import json
import time
import math
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta, timezone
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import csv

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "backtest_smc_v3.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
BACKTEST_DIR = Path("backtests")
BACKTEST_DIR.mkdir(exist_ok=True)
BULLISH_LEG = 1
BEARISH_LEG = 0


class MarketBias(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


class StructureType(Enum):
    BOS = "BOS"
    CHOCH = "CHoCH"


class OrderBlockType(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"


class PivotType(Enum):
    HIGH = "high"
    LOW = "low"


class SwingType(Enum):
    HH = "HH"
    LH = "LH"
    HL = "HL"
    LL = "LL"


@dataclass
class Pivot:
    level: float
    time: int
    index: int
    type: PivotType
    swing_type: Optional[SwingType] = None
    is_internal: bool = False
    crossed: bool = False


@dataclass
class Structure:
    type: StructureType
    bias: MarketBias
    level: float
    time: int
    pivot: Pivot
    is_internal: bool = False


@dataclass
class OrderBlock:
    type: OrderBlockType
    top: float
    bottom: float
    time: int
    index: int = 0
    mitigated: bool = False
    is_internal: bool = False
    volume: float = 0.0
    strength: float = 0.0
    mitigation_level: float = 0.0


@dataclass
class FairValueGap:
    type: OrderBlockType
    top: float
    bottom: float
    time: int
    filled: bool = False
    fill_percentage: float = 0.0


@dataclass
class LiquidityPool:
    level: float
    type: str
    touches: List[int]
    strength: int
    swept: bool = False


@dataclass
class TrailingExtremes:
    top: float = -float("inf")
    bottom: float = float("inf")
    top_time: int = 0
    bottom_time: int = 0
    top_index: int = 0
    bottom_index: int = 0


@dataclass
class MultiTimeframeAnalysis:
    daily_bias: MarketBias = MarketBias.NEUTRAL
    weekly_bias: MarketBias = MarketBias.NEUTRAL
    daily_support: Optional[float] = None
    daily_resistance: Optional[float] = None
    weekly_support: Optional[float] = None
    weekly_resistance: Optional[float] = None
    volume_profile: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradeSimulation:
    trade_id: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    decision: str
    stop_loss: float
    take_profit: float
    leverage: int
    pnl: float
    pnl_pct: float
    holding_period: timedelta
    commission: float
    slippage: float
    reason: str
    smc_context: Dict[str, Any] = field(default_factory=dict)
    entry_confidence: float = 0.0
    exit_confidence: float = 0.0
    risk_amount: float = 0.0
    risk_reward_ratio: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "decision": self.decision,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "leverage": self.leverage,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "holding_period": str(self.holding_period),
            "commission": self.commission,
            "slippage": self.slippage,
            "reason": self.reason,
            "entry_confidence": self.entry_confidence,
            "exit_confidence": self.exit_confidence,
            "risk_amount": self.risk_amount,
            "risk_reward_ratio": self.risk_reward_ratio,
            "max_adverse_excursion": self.max_adverse_excursion,
            "max_favorable_excursion": self.max_favorable_excursion,
            "smc_context": {
                "setup_quality": self.smc_context.get("setup_quality", "N/A"),
                "confluence_score": self.smc_context.get("confluence_score", 0.0),
                "liquidity_sweep": self.smc_context.get("liquidity_sweep", False),
                "risk_reward": self.smc_context.get("risk_reward", 0.0),
                "liquidity_pools": [
                    {
                        "level": float(lp.get("level", 0)),
                        "type": lp.get("type", ""),
                        "swept": bool(lp.get("swept", False)),
                    }
                    for lp in self.smc_context.get("liquidity_pools", [])[:3]
                ],
            },
        }


@dataclass
class BacktestResult:
    symbol: str
    interval: str
    start_time: datetime
    end_time: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    annualized_return: float
    total_trades: int
    winning_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    recovery_factor: float
    avg_trade_duration: timedelta
    avg_profit_per_trade: float
    avg_loss_per_trade: float
    largest_winning_trade: float
    largest_losing_trade: float
    trades: List[TradeSimulation]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    best_trade: Optional[TradeSimulation] = None
    worst_trade: Optional[TradeSimulation] = None
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    trade_distribution: Dict[str, List[float]] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    smc_stats: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, file_path: Path) -> None:
        data = {
            "symbol": self.symbol,
            "interval": self.interval,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "recovery_factor": self.recovery_factor,
            "avg_trade_duration": str(self.avg_trade_duration)
            if hasattr(self, "avg_trade_duration")
            else "0 days",
            "avg_profit_per_trade": getattr(self, "avg_profit_per_trade", 0.0),
            "avg_loss_per_trade": getattr(self, "avg_loss_per_trade", 0.0),
            "largest_winning_trade": getattr(self, "largest_winning_trade", 0.0),
            "largest_losing_trade": getattr(self, "largest_losing_trade", 0.0),
            "best_trade": self.best_trade.to_dict() if self.best_trade else None,
            "worst_trade": self.worst_trade.to_dict() if self.worst_trade else None,
            "monthly_returns": self.monthly_returns
            if hasattr(self, "monthly_returns")
            else {},
            "trade_distribution": self.trade_distribution
            if hasattr(self, "trade_distribution")
            else {},
            "risk_metrics": self.risk_metrics if hasattr(self, "risk_metrics") else {},
            "smc_stats": self.smc_stats if hasattr(self, "smc_stats") else {},
        }
        data["trades"] = (
            [trade.to_dict() for trade in self.trades] if self.trades else []
        )
        if not self.equity_curve.empty:
            data["equity_curve"] = {
                "values": self.equity_curve.values.tolist(),
                "index": [idx.isoformat() for idx in self.equity_curve.index],
            }
        if not self.drawdown_curve.empty:
            data["drawdown_curve"] = {
                "values": self.drawdown_curve.values.tolist(),
                "index": [idx.isoformat() for idx in self.drawdown_curve.index],
            }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if len(df) < period + 1:
        return pd.Series([0.01 * df["close"].iloc[-1]] * len(df), index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    atr = atr.ffill().bfill()
    if atr.isna().any():
        atr = atr.fillna(0.01 * close)
    return atr


def calculate_volume_profile(df: pd.DataFrame, window: int = 50) -> Dict[str, Any]:
    if len(df) < window:
        return {"high_volume_nodes": [], "low_volume_nodes": []}
    recent = df.tail(window)
    price_range = recent["high"].max() - recent["low"].min()
    if price_range == 0:
        return {"high_volume_nodes": [], "low_volume_nodes": []}
    bin_count = max(5, min(20, window // 5))
    bins = np.linspace(recent["low"].min(), recent["high"].max(), bin_count + 1)
    volume_profile = {}
    for i in range(len(bins) - 1):
        bin_mask = (recent["close"] >= bins[i]) & (recent["close"] < bins[i + 1])
        bin_volume = recent.loc[bin_mask, "volume"].sum()
        bin_price = (bins[i] + bins[i + 1]) / 2
        volume_profile[bin_price] = bin_volume
    sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
    high_volume_threshold = sorted_profile[0][1] * 0.7 if sorted_profile else 0
    high_volume_nodes = [
        price for (price, vol) in sorted_profile if vol >= high_volume_threshold
    ]
    low_volume_nodes = [
        price for (price, vol) in sorted_profile if vol < high_volume_threshold * 0.3
    ]
    return {
        "high_volume_nodes": high_volume_nodes[:3],
        "low_volume_nodes": low_volume_nodes[:3],
    }


class BinanceDataClient:
    BASE_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"

    def __init__(self, use_futures: bool = True):
        self.use_futures = use_futures
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self._market_info_cache = {}

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{(self.FUTURES_URL if self.use_futures else self.BASE_URL)}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
                response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = f"Binance API error: {e}"
            if hasattr(e, "response") and e.response is not None:
                error_msg += f" | Response: {e.response.text}"
            logger.error(error_msg)
            raise

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        endpoint = "/fapi/v1/klines" if self.use_futures else "/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500 if self.use_futures else 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        all_data = []
        current_start = start_time
        max_limit = 1500 if self.use_futures else 1000
        while True:
            if current_start and len(all_data) > 0:
                params["startTime"] = current_start
            data = self._get(endpoint, params)
            if not data or len(data) == 0:
                break
            all_data.extend(data)
            if len(data) < params["limit"]:
                break
            last_timestamp = data[-1][0]
            current_start = last_timestamp + 1
            if len(all_data) >= limit:
                all_data = all_data[:limit]
                break
        if not all_data:
            logger.warning(f"No klines data returned for {symbol} {interval}")
            return pd.DataFrame()
        df = pd.DataFrame(
            all_data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = df[
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        ]
        if self.use_futures:
            df["volume"] = df["volume"] * df["close"]
        df = df[~df.index.duplicated(keep="first")]
        logger.info(
            f"✅ Получено {len(df)} валидных свечей для {symbol} на интервале {interval}"
        )
        return df

    def get_24h_ticker(self, symbol: str) -> Dict[str, Any]:
        if (
            symbol in self._market_info_cache
            and time.time() - self._market_info_cache[symbol]["time"] < 60
        ):
            return self._market_info_cache[symbol]["data"]
        endpoint = "/fapi/v1/ticker/24hr" if self.use_futures else "/api/v3/ticker/24hr"
        params = {"symbol": symbol}
        try:
            data = self._get(endpoint, params)
            self._market_info_cache[symbol] = {"time": time.time(), "data": data}
            return data
        except Exception as e:
            logger.error(f"Error getting 24h ticker for {symbol}: {e}")
            return {}


class SmartMoneyConceptsAnalyzer:
    def __init__(
        self,
        lookback: int = 50,
        internal_confluence_filter: bool = False,
        volatility_method: str = "ATR",
        atr_length: int = 200,
        ob_mitigation: str = "HIGHLOW",
        min_volume_threshold: float = 100000.0,
    ):
        self.lookback = lookback
        self.internal_confluence_filter = internal_confluence_filter
        self.volatility_method = volatility_method
        self.atr_length = atr_length
        self.ob_mitigation = ob_mitigation
        self.min_volume_threshold = min_volume_threshold
        self.swing_bias = MarketBias.NEUTRAL
        self.internal_bias = MarketBias.NEUTRAL
        self.confirmed_swing_highs: List[Pivot] = []
        self.confirmed_swing_lows: List[Pivot] = []
        self.swing_highs: List[Pivot] = []
        self.swing_lows: List[Pivot] = []
        self.last_analyzed_time = 0
        self._last_result = None
        self.trailing = TrailingExtremes()
        self.parsed_highs: List[float] = []
        self.parsed_lows: List[float] = []
        self.times: List[int] = []
        self.volume_profile: Dict[str, Any] = {}
        self.crossed_swing_highs: set = set()
        self.crossed_swing_lows: set = set()
        self.crossed_internal_highs: set = set()
        self.crossed_internal_lows: set = set()
        self.last_swing_pivot_index = 0

    def find_fvgs(self, df: pd.DataFrame) -> List[FairValueGap]:
        fvgs = []
        if len(df) < 3:
            return fvgs
        curr_low = df["low"].iloc[-1]
        curr_high = df["high"].iloc[-1]
        last2_high = df["high"].iloc[-3]
        last2_low = df["low"].iloc[-3]
        if curr_low > last2_high:
            fvgs.append(
                FairValueGap(
                    type=OrderBlockType.BULLISH,
                    top=float(curr_low),
                    bottom=float(last2_high),
                    time=int(df.index[-1].timestamp()),
                )
            )
        if curr_high < last2_low:
            fvgs.append(
                FairValueGap(
                    type=OrderBlockType.BEARISH,
                    top=float(last2_low),
                    bottom=float(curr_high),
                    time=int(df.index[-1].timestamp()),
                )
            )
        return fvgs

    def _create_order_block(
        self,
        df: pd.DataFrame,
        pivot: Pivot,
        break_index: int,
        bias: MarketBias,
        internal: bool,
    ) -> Optional[OrderBlock]:
        if pivot.index >= break_index:
            return None
        slice_df = df.iloc[pivot.index : break_index + 1]
        if slice_df.empty:
            return None
        if bias == MarketBias.BULLISH:
            min_idx_rel = slice_df["low"].argmin()
            ob_candle = slice_df.iloc[min_idx_rel]
            ob_index = pivot.index + min_idx_rel
            return OrderBlock(
                type=OrderBlockType.BULLISH,
                top=float(ob_candle["high"]),
                bottom=float(ob_candle["low"]),
                time=int(ob_candle.name.timestamp()),
                index=ob_index,
                is_internal=internal,
                mitigated=False,
            )
        else:
            max_idx_rel = slice_df["high"].argmax()
            ob_candle = slice_df.iloc[max_idx_rel]
            ob_index = pivot.index + max_idx_rel
            return OrderBlock(
                type=OrderBlockType.BEARISH,
                top=float(ob_candle["high"]),
                bottom=float(ob_candle["low"]),
                time=int(ob_candle.name.timestamp()),
                index=ob_index,
                is_internal=internal,
                mitigated=False,
            )

    def _calculate_leg(self, df: pd.DataFrame, size: int) -> pd.Series:
        leg_arr = np.zeros(len(df), dtype=int)
        highs = df["high"].values
        lows = df["low"].values
        for i in range(size, len(df)):
            window_start = i - size
            window_end = i
            if window_start < 0:
                continue
            window_highs = highs[window_start:window_end]
            window_lows = lows[window_start:window_end]
            if len(window_highs) > 0 and len(window_lows) > 0:
                if highs[i] > np.max(window_highs):
                    leg_arr[i] = BEARISH_LEG
                elif lows[i] < np.min(window_lows):
                    leg_arr[i] = BULLISH_LEG
        return pd.Series(leg_arr, index=df.index)

    def _prepare_parsed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if self.volatility_method == "RANGE":
            tr = df_copy["high"] - df_copy["low"]
            volatility_measure = tr.expanding().mean()
        else:
            volatility_measure = compute_atr(df_copy, self.atr_length)
        avg_volume = df_copy["volume"].rolling(20).mean()
        high_volume_mask = df_copy["volume"] > avg_volume * 1.5
        high_volatility_bar = df_copy["high"] - df_copy["low"] >= 2 * volatility_measure
        df_copy["parsed_high"] = np.where(
            high_volatility_bar & high_volume_mask, df_copy["low"], df_copy["high"]
        )
        df_copy["parsed_low"] = np.where(
            high_volatility_bar & high_volume_mask, df_copy["high"], df_copy["low"]
        )
        self.parsed_highs = df_copy["parsed_high"].tolist()
        self.parsed_lows = df_copy["parsed_low"].tolist()
        self.times = [int(ts.timestamp()) for ts in df_copy.index]
        self.volume_profile = calculate_volume_profile(df_copy, window=100)
        return df_copy

    def find_pivots_luxalgo(
        self, df: pd.DataFrame, size: int = 10, internal: bool = False
    ) -> Tuple[List[Pivot], List[Pivot]]:
        highs = df["high"].values
        lows = df["low"].values
        times = df.index.astype(np.int64) // 10**9
        pivot_highs = []
        pivot_lows = []
        for i in range(size, len(df) - size):
            window_highs = highs[i - size : i + size + 1]
            window_lows = lows[i - size : i + size + 1]
            if len(window_highs) < size * 2 + 1:
                continue
            if highs[i] == np.max(window_highs):
                pivot_highs.append(
                    Pivot(
                        level=float(highs[i]),
                        time=int(times[i]),
                        index=i,
                        type=PivotType.HIGH,
                        is_internal=internal,
                        crossed=False,
                    )
                )
            if lows[i] == np.min(window_lows):
                pivot_lows.append(
                    Pivot(
                        level=float(lows[i]),
                        time=int(times[i]),
                        index=i,
                        type=PivotType.LOW,
                        is_internal=internal,
                        crossed=False,
                    )
                )
        for k in range(1, len(pivot_highs)):
            curr = pivot_highs[k]
            prev = pivot_highs[k - 1]
            curr.swing_type = SwingType.HH if curr.level > prev.level else SwingType.LH
        for k in range(1, len(pivot_lows)):
            curr = pivot_lows[k]
            prev = pivot_lows[k - 1]
            curr.swing_type = SwingType.HL if curr.level > prev.level else SwingType.LL
        return (pivot_highs, pivot_lows)

    def _is_significant_high(self, df: pd.DataFrame, index: int, window: int) -> bool:
        if index < window or index >= len(df) - window:
            return True
        current_high = df["high"].iloc[index]
        left_window = df["high"].iloc[max(0, index - window) : index]
        right_window = df["high"].iloc[index + 1 : min(len(df), index + window + 1)]
        if not (current_high > left_window.max() and current_high > right_window.max()):
            return False
        current_volume = df["volume"].iloc[index]
        avg_volume = (
            df["volume"].iloc[max(0, index - 5) : min(len(df), index + 5)].mean()
        )
        return current_volume > avg_volume * 0.8

    def _is_significant_low(self, df: pd.DataFrame, index: int, window: int) -> bool:
        if index < window or index >= len(df) - window:
            return True
        current_low = df["low"].iloc[index]
        left_window = df["low"].iloc[max(0, index - window) : index]
        right_window = df["low"].iloc[index + 1 : min(len(df), index + window + 1)]
        if not (current_low < left_window.min() and current_low < right_window.min()):
            return False
        current_volume = df["volume"].iloc[index]
        avg_volume = (
            df["volume"].iloc[max(0, index - 5) : min(len(df), index + 5)].mean()
        )
        return current_volume > avg_volume * 0.8

    def _calculate_confluence_bars(
        self, df: pd.DataFrame
    ) -> Tuple[List[bool], List[bool]]:
        bullish_bars = []
        bearish_bars = []
        for i in range(len(df)):
            open_price = df["open"].iloc[i]
            high_price = df["high"].iloc[i]
            low_price = df["low"].iloc[i]
            close_price = df["close"].iloc[i]
            body_top = max(open_price, close_price)
            body_bottom = min(open_price, close_price)
            upper_wick = high_price - body_top
            lower_wick = body_bottom - low_price
            bullish_bars.append(
                upper_wick < lower_wick * 0.7 and close_price > open_price
            )
            bearish_bars.append(
                lower_wick < upper_wick * 0.7 and close_price < open_price
            )
        return (bullish_bars, bearish_bars)

    def detect_structure_break(
        self,
        df: pd.DataFrame,
        pivots_high: List[Pivot],
        pivots_low: List[Pivot],
        internal: bool = False,
    ) -> List[Structure]:
        structures = []
        current_bias = self.internal_bias if internal else self.swing_bias
        atr = compute_atr(df, self.atr_length)
        (bullish_bars, bearish_bars) = ([], [])
        if internal and self.internal_confluence_filter:
            (bullish_bars, bearish_bars) = self._calculate_confluence_bars(df)
        if internal:
            crossed_highs = self.crossed_internal_highs
            crossed_lows = self.crossed_internal_lows
        else:
            crossed_highs = self.crossed_swing_highs
            crossed_lows = self.crossed_swing_lows

        def is_different_from_swing(pivot: Pivot) -> bool:
            if not internal:
                return True
            atr_value = (
                atr.iloc[pivot.index] if pivot.index < len(atr) else atr.iloc[-1]
            )
            threshold = (
                atr_value * 0.1
                if pd.notna(atr_value)
                else 0.01 * df["close"].iloc[pivot.index]
            )
            swing_pivots = (
                self.swing_highs if pivot.type == PivotType.HIGH else self.swing_lows
            )
            for swing_pivot in swing_pivots:
                if (
                    abs(pivot.level - swing_pivot.level) < threshold
                    and abs(pivot.index - swing_pivot.index) <= 3
                ):
                    return False
            return True

        unique_highs = self._filter_duplicate_pivots(pivots_high, atr)
        unique_lows = self._filter_duplicate_pivots(pivots_low, atr)
        for i in range(1, len(df)):
            if i < 1:
                continue
            close = df["close"].iloc[i]
            prev_close = df["close"].iloc[i - 1]
            current_volume = df["volume"].iloc[i]
            avg_volume = df["volume"].iloc[max(0, i - 20) : i].mean()
            if current_volume < avg_volume * 0.5:
                continue
            structure_found = False
            if not structure_found and unique_highs:
                for pivot in sorted(unique_highs, key=lambda x: x.index, reverse=True):
                    if pivot.index >= i or pivot.index in crossed_highs:
                        continue
                    if prev_close <= pivot.level and close > pivot.level:
                        if internal:
                            if not is_different_from_swing(pivot):
                                continue
                            if (
                                self.internal_confluence_filter
                                and i < len(bullish_bars)
                                and (not bullish_bars[i])
                            ):
                                continue
                        breakout_volume = (
                            df["volume"].iloc[max(0, i - 3) : i + 1].mean()
                        )
                        if breakout_volume < avg_volume * 1.2:
                            continue
                        struct_type = (
                            StructureType.CHOCH
                            if current_bias == MarketBias.BEARISH
                            else StructureType.BOS
                        )
                        current_bias = MarketBias.BULLISH
                        structures.append(
                            Structure(
                                type=struct_type,
                                bias=current_bias,
                                level=pivot.level,
                                time=int(df.index[i].timestamp()),
                                pivot=pivot,
                                is_internal=internal,
                            )
                        )
                        crossed_highs.add(pivot.index)
                        structure_found = True
                        if internal:
                            self.internal_bias = current_bias
                        else:
                            self.swing_bias = current_bias
                        self.last_swing_pivot_index = i
                        break
            if not structure_found and unique_lows:
                for pivot in sorted(unique_lows, key=lambda x: x.index, reverse=True):
                    if pivot.index >= i or pivot.index in crossed_lows:
                        continue
                    if prev_close >= pivot.level and close < pivot.level:
                        if internal:
                            if not is_different_from_swing(pivot):
                                continue
                            if (
                                self.internal_confluence_filter
                                and i < len(bearish_bars)
                                and (not bearish_bars[i])
                            ):
                                continue
                        breakout_volume = (
                            df["volume"].iloc[max(0, i - 3) : i + 1].mean()
                        )
                        if breakout_volume < avg_volume * 1.2:
                            continue
                        struct_type = (
                            StructureType.CHOCH
                            if current_bias == MarketBias.BULLISH
                            else StructureType.BOS
                        )
                        current_bias = MarketBias.BEARISH
                        structures.append(
                            Structure(
                                type=struct_type,
                                bias=current_bias,
                                level=pivot.level,
                                time=int(df.index[i].timestamp()),
                                pivot=pivot,
                                is_internal=internal,
                            )
                        )
                        crossed_lows.add(pivot.index)
                        structure_found = True
                        if internal:
                            self.internal_bias = current_bias
                        else:
                            self.swing_bias = current_bias
                        self.last_swing_pivot_index = i
                        break
        return structures

    def _filter_duplicate_pivots(
        self, pivots: List[Pivot], atr: pd.Series
    ) -> List[Pivot]:
        if not pivots:
            return []
        filtered = []
        used_levels = []
        for pivot in sorted(pivots, key=lambda x: x.index):
            current_atr = (
                atr.iloc[pivot.index] if pivot.index < len(atr) else atr.iloc[-1]
            )
            threshold = current_atr * 0.15
            is_duplicate = False
            for level in used_levels:
                if abs(pivot.level - level) < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(pivot)
                used_levels.append(pivot.level)
        return filtered

    def find_order_blocks_luxalgo(
        self, df: pd.DataFrame, structure: Structure, internal: bool = False
    ) -> Optional[OrderBlock]:
        if not self.parsed_highs or not self.parsed_lows:
            return None
        pivot_idx = structure.pivot.index
        try:
            break_idx = next(
                (i for (i, t) in enumerate(self.times) if t == structure.time)
            )
        except StopIteration:
            return None
        if pivot_idx >= break_idx or pivot_idx < 0:
            return None
        candidates: List[OrderBlock] = []
        try:
            if structure.bias == MarketBias.BULLISH:
                for i in range(break_idx - 1, pivot_idx - 1, -1):
                    open_price = float(df["open"].iloc[i])
                    close_price = float(df["close"].iloc[i])
                    if close_price < open_price:
                        if (
                            i + 1 < len(df)
                            and float(df["close"].iloc[i + 1]) > open_price
                        ):
                            top = float(df["high"].iloc[i])
                            bottom = float(df["low"].iloc[i])
                            vol = float(df["volume"].iloc[i])
                            mitigation = (
                                float(df["high"].iloc[i])
                                if self.ob_mitigation == "HIGHLOW"
                                else float(close_price)
                            )
                            strength = min(
                                1.0,
                                (
                                    vol
                                    / (
                                        df["volume"].iloc[max(0, i - 10) : i + 1].mean()
                                        + 1e-09
                                    )
                                    + abs(close_price - open_price)
                                    / max(
                                        1e-09,
                                        float(df["high"].iloc[i] - df["low"].iloc[i]),
                                    )
                                )
                                / 2,
                            )
                            candidates.append(
                                OrderBlock(
                                    type=OrderBlockType.BULLISH,
                                    top=top,
                                    bottom=bottom,
                                    time=int(df.index[i].timestamp()),
                                    volume=vol,
                                    strength=strength,
                                    is_internal=internal,
                                    mitigation_level=mitigation,
                                )
                            )
                            break
            else:
                for i in range(break_idx - 1, pivot_idx - 1, -1):
                    open_price = float(df["open"].iloc[i])
                    close_price = float(df["close"].iloc[i])
                    if close_price > open_price:
                        if (
                            i + 1 < len(df)
                            and float(df["close"].iloc[i + 1]) < open_price
                        ):
                            top = float(df["high"].iloc[i])
                            bottom = float(df["low"].iloc[i])
                            vol = float(df["volume"].iloc[i])
                            mitigation = (
                                float(df["low"].iloc[i])
                                if self.ob_mitigation == "HIGHLOW"
                                else float(close_price)
                            )
                            strength = min(
                                1.0,
                                (
                                    vol
                                    / (
                                        df["volume"].iloc[max(0, i - 10) : i + 1].mean()
                                        + 1e-09
                                    )
                                    + abs(close_price - open_price)
                                    / max(
                                        1e-09,
                                        float(df["high"].iloc[i] - df["low"].iloc[i]),
                                    )
                                )
                                / 2,
                            )
                            candidates.append(
                                OrderBlock(
                                    type=OrderBlockType.BEARISH,
                                    top=top,
                                    bottom=bottom,
                                    time=int(df.index[i].timestamp()),
                                    volume=vol,
                                    strength=strength,
                                    is_internal=internal,
                                    mitigation_level=mitigation,
                                )
                            )
                            break
            return candidates[0] if candidates else None
        except Exception as e:
            logger.warning(f"Error finding order block (new method): {e}")
            return None

    def update_order_blocks_mitigation(
        self, df: pd.DataFrame, order_blocks: List[OrderBlock], use_close: bool = None
    ) -> List[OrderBlock]:
        active_blocks = []
        current_price = df["close"].iloc[-1]
        current_high = df["high"].iloc[-1]
        current_low = df["low"].iloc[-1]
        if use_close is None:
            use_close = self.ob_mitigation == "CLOSE"
        for ob in order_blocks:
            if ob.mitigated:
                continue
            try:
                if ob.type == OrderBlockType.BULLISH:
                    source = current_price if use_close else current_low
                    if source < (ob.mitigation_level or ob.bottom):
                        ob.mitigated = True
                    else:
                        active_blocks.append(ob)
                else:
                    source = current_price if use_close else current_high
                    if source > (ob.mitigation_level or ob.top):
                        ob.mitigated = True
                    else:
                        active_blocks.append(ob)
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning(f"Error updating order block: {e}")
                continue
        return active_blocks

    def find_fair_value_gaps(
        self, df: pd.DataFrame, min_gap_size: float = 0.001, auto_threshold: bool = True
    ) -> List[FairValueGap]:
        gaps = []
        if len(df) < 3:
            return gaps
        if auto_threshold:
            returns = df["close"].pct_change().abs()
            threshold = (
                returns.rolling(20).mean().iloc[-1] * 2 if len(returns) > 20 else 0.001
            )
            if pd.notna(threshold):
                min_gap_size = max(min_gap_size, float(threshold))
        for i in range(2, len(df)):
            current_low = df["low"].iloc[i]
            current_high = df["high"].iloc[i]
            last2_high = df["high"].iloc[i - 2]
            last2_low = df["low"].iloc[i - 2]
            last_close = df["close"].iloc[i - 1]
            avg_volume = df["volume"].iloc[max(0, i - 5) : i + 1].mean()
            current_volume = df["volume"].iloc[i]
            if current_volume < avg_volume * 0.7:
                continue
            if current_low > last2_high and last_close > last2_high:
                gap_size = current_low - last2_high
                if gap_size > min_gap_size * df["close"].iloc[i]:
                    gaps.append(
                        FairValueGap(
                            type=OrderBlockType.BULLISH,
                            top=float(current_low),
                            bottom=float(last2_high),
                            time=int(df.index[i - 1].timestamp()),
                        )
                    )
            elif current_high < last2_low and last_close < last2_low:
                gap_size = last2_low - current_high
                if gap_size > min_gap_size * df["close"].iloc[i]:
                    gaps.append(
                        FairValueGap(
                            type=OrderBlockType.BEARISH,
                            top=float(last2_low),
                            bottom=float(current_high),
                            time=int(df.index[i - 1].timestamp()),
                        )
                    )
        return self._filter_duplicate_fvgs(gaps, df)

    def _filter_duplicate_fvgs(
        self, fvgs: List[FairValueGap], df: pd.DataFrame
    ) -> List[FairValueGap]:
        if not fvgs:
            return []
        filtered = []
        used_levels = []
        current_price = df["close"].iloc[-1]
        for fvg in sorted(fvgs, key=lambda x: x.time, reverse=True):
            threshold = 0.01 * current_price
            is_duplicate = False
            for level in used_levels:
                if abs((fvg.top + fvg.bottom) / 2 - level) < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(fvg)
                used_levels.append((fvg.top + fvg.bottom) / 2)
        return filtered[:5]

    def update_fair_value_gaps(
        self, df: pd.DataFrame, gaps: List[FairValueGap]
    ) -> List[FairValueGap]:
        active_gaps = []
        current_high = df["high"].iloc[-1]
        current_low = df["low"].iloc[-1]
        for gap in gaps:
            if gap.filled:
                continue
            try:
                if gap.type == OrderBlockType.BULLISH:
                    if current_low < gap.bottom:
                        gap.filled = True
                        gap.fill_percentage = 100.0
                    elif current_low < gap.top:
                        gap.fill_percentage = float(
                            (gap.top - current_low)
                            / max(1e-09, gap.top - gap.bottom)
                            * 100
                        )
                        active_gaps.append(gap)
                    else:
                        active_gaps.append(gap)
                elif current_high > gap.top:
                    gap.filled = True
                    gap.fill_percentage = 100.0
                elif current_high > gap.bottom:
                    gap.fill_percentage = float(
                        (current_high - gap.bottom)
                        / max(1e-09, gap.top - gap.bottom)
                        * 100
                    )
                    active_gaps.append(gap)
                else:
                    active_gaps.append(gap)
            except (TypeError, ValueError, ZeroDivisionError, AttributeError) as e:
                logger.warning(f"Error updating FVG: {e}")
                continue
        return active_gaps

    def find_liquidity_pools(
        self, df: pd.DataFrame, min_touches: int = 2
    ) -> List[LiquidityPool]:
        if len(df) < min_touches * 2:
            return []
        pools: List[LiquidityPool] = []
        atr = compute_atr(df, 14)
        avg_atr = (
            atr.mean() if atr.notna().any() else (df["high"] - df["low"]).mean() * 0.1
        )
        tol = (
            float(avg_atr * 0.15)
            if pd.notna(avg_atr)
            else float((df["high"] - df["low"]).mean() * 0.1)
        )
        volume_profile = calculate_volume_profile(df, window=200)
        high_volume_nodes = volume_profile["high_volume_nodes"]
        high_levels: Dict[float, Dict[str, Any]] = {}
        for i in range(len(df)):
            high = float(df["high"].iloc[i])
            volume = float(df["volume"].iloc[i])
            near_high_volume = any(
                (abs(high - node) < tol for node in high_volume_nodes)
            )
            found = False
            for level in list(high_levels.keys()):
                if abs(high - level) < tol:
                    high_levels[level]["touches"].append(i)
                    high_levels[level]["count"] += 1
                    high_levels[level]["volume"] += volume
                    found = True
                    break
            if not found and (near_high_volume or volume > df["volume"].mean()):
                high_levels[high] = {"touches": [i], "count": 1, "volume": volume}
        for level, data in high_levels.items():
            if (
                data["count"] >= min_touches
                and data["volume"] > df["volume"].mean() * min_touches
            ):
                pools.append(
                    LiquidityPool(
                        level=float(level),
                        type="EQH",
                        touches=data["touches"],
                        strength=int(data["count"]),
                    )
                )
        low_levels: Dict[float, Dict[str, Any]] = {}
        for i in range(len(df)):
            low = float(df["low"].iloc[i])
            volume = float(df["volume"].iloc[i])
            near_high_volume = any(
                (abs(low - node) < tol for node in high_volume_nodes)
            )
            found = False
            for level in list(low_levels.keys()):
                if abs(low - level) < tol:
                    low_levels[level]["touches"].append(i)
                    low_levels[level]["count"] += 1
                    low_levels[level]["volume"] += volume
                    found = True
                    break
            if not found and (near_high_volume or volume > df["volume"].mean()):
                low_levels[low] = {"touches": [i], "count": 1, "volume": volume}
        for level, data in low_levels.items():
            if (
                data["count"] >= min_touches
                and data["volume"] > df["volume"].mean() * min_touches
            ):
                pools.append(
                    LiquidityPool(
                        level=float(level),
                        type="EQL",
                        touches=data["touches"],
                        strength=int(data["count"]),
                    )
                )
        pools.sort(key=lambda x: (x.strength, x.level), reverse=True)
        return pools[:15]

    def update_liquidity_pools(
        self, df: pd.DataFrame, pools: List[LiquidityPool]
    ) -> List[LiquidityPool]:
        active = []
        current_high = df["high"].iloc[-1]
        current_low = df["low"].iloc[-1]
        for p in pools:
            if p.swept:
                continue
            try:
                if p.type == "EQH":
                    if current_high > p.level * 1.002:
                        p.swept = True
                    else:
                        active.append(p)
                elif current_low < p.level * 0.998:
                    p.swept = True
                else:
                    active.append(p)
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning(f"Error updating liquidity pool: {e}")
                continue
        return active

    def update_trailing_extremes(self, df: pd.DataFrame):
        if len(df) == 0:
            return
        if self.trailing.top == -float("inf") or self.trailing.bottom == float("inf"):
            recent = df.tail(200) if len(df) > 200 else df
            self.trailing.top = float(recent["high"].max())
            self.trailing.bottom = float(recent["low"].min())
            self.trailing.top_time = (
                int(recent.index[-1].timestamp()) if len(recent) > 0 else 0
            )
            self.trailing.bottom_time = (
                int(recent.index[-1].timestamp()) if len(recent) > 0 else 0
            )
            self.trailing.top_index = len(df) - 1
            self.trailing.bottom_index = len(df) - 1
            return
        current_high = df["high"].iloc[-1]
        current_low = df["low"].iloc[-1]
        current_time = int(df.index[-1].timestamp())
        current_index = len(df) - 1
        current_volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].tail(20).mean()
        if current_high > self.trailing.top and current_volume > avg_volume * 0.8:
            self.trailing.top = current_high
            self.trailing.top_time = current_time
            self.trailing.top_index = current_index
        if current_low < self.trailing.bottom and current_volume > avg_volume * 0.8:
            self.trailing.bottom = current_low
            self.trailing.bottom_time = current_time
            self.trailing.bottom_index = current_index

    def get_premium_discount_zones(
        self, df: pd.DataFrame
    ) -> Dict[str, Tuple[float, float]]:
        self.update_trailing_extremes(df)
        hi = self.trailing.top
        lo = self.trailing.bottom
        if hi == -float("inf") or lo == float("inf"):
            recent = df.tail(200) if len(df) > 200 else df
            hi = float(recent["high"].max()) if len(recent) > 0 else 0
            lo = float(recent["low"].min()) if len(recent) > 0 else 0
            self.trailing.top = hi
            self.trailing.bottom = lo
        if hi <= lo:
            mid = df["close"].iloc[-1] if len(df) > 0 else 100.0
            return {
                "premium": (mid, mid),
                "equilibrium": (mid, mid),
                "discount": (mid, mid),
            }
        atr_series = compute_atr(df, 20)
        atr = atr_series.iloc[-1] if len(atr_series) > 0 else 0
        volatility_ratio = atr / ((hi + lo) / 2) if hi + lo > 0 else 0.01
        premium_factor_top = max(0.7, min(0.98, 0.85 + volatility_ratio * 5))
        discount_factor_bottom = min(0.3, max(0.02, 0.15 - volatility_ratio * 5))
        equilibrium_top_factor = 0.525 - volatility_ratio * 0.05
        equilibrium_bottom_factor = 0.475 + volatility_ratio * 0.05
        premium_bottom = lo + (hi - lo) * premium_factor_top
        discount_top = lo + (hi - lo) * discount_factor_bottom
        equilibrium_top = lo + (hi - lo) * equilibrium_top_factor
        equilibrium_bottom = lo + (hi - lo) * equilibrium_bottom_factor
        if premium_bottom < equilibrium_top:
            premium_bottom = equilibrium_top
        if discount_top > equilibrium_bottom:
            discount_top = equilibrium_bottom
        zones = {
            "premium": (premium_bottom, hi),
            "equilibrium": (equilibrium_bottom, equilibrium_top),
            "discount": (lo, discount_top),
        }
        return zones

    def detect_liquidity_sweeps(
        self, df: pd.DataFrame, smc_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        current_price = smc_data["current_price"]
        last_swing_high = smc_data.get("last_swing_high")
        last_swing_low = smc_data.get("last_swing_low")
        if not last_swing_high or not last_swing_low:
            return {
                "bullish_sweep": False,
                "bearish_sweep": False,
                "distance": 0.0,
                "valid": False,
            }
        atr_series = compute_atr(df, 14)
        atr = atr_series.iloc[-1] if len(atr_series) > 0 else 0.01 * current_price
        atr = max(atr, 0.001 * current_price)
        current_volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].tail(20).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        bullish_sweep = False
        bearish_sweep = False
        sweep_distance = 0.0
        if last_swing_low and current_price < last_swing_low:
            distance = last_swing_low - current_price
            if distance > 0 and distance <= 2.0 * atr:
                if len(df) > 2:
                    prev_price = df["close"].iloc[-2]
                    if (
                        current_price > prev_price
                        and current_price > df["open"].iloc[-1]
                        and (volume_ratio > 0.8)
                    ):
                        bullish_sweep = True
                        sweep_distance = distance
        if last_swing_high and current_price > last_swing_high:
            distance = current_price - last_swing_high
            if distance > 0 and distance <= 2.0 * atr:
                if len(df) > 2:
                    prev_price = df["close"].iloc[-2]
                    if (
                        current_price < prev_price
                        and current_price < df["open"].iloc[-1]
                        and (volume_ratio > 0.8)
                    ):
                        bearish_sweep = True
                        sweep_distance = distance
        return {
            "bullish_sweep": bullish_sweep,
            "bearish_sweep": bearish_sweep,
            "sweep_distance": float(sweep_distance),
            "valid": bullish_sweep or bearish_sweep,
        }

    def get_strong_weak_levels(self) -> Dict[str, str]:
        if self.swing_bias == MarketBias.BULLISH:
            return {"high": "Weak High", "low": "Strong Low"}
        elif self.swing_bias == MarketBias.BEARISH:
            return {"high": "Strong High", "low": "Weak Low"}
        else:
            return {"high": "Neutral High", "low": "Neutral Low"}

    def get_multi_timeframe_analysis(self, df: pd.DataFrame) -> MultiTimeframeAnalysis:
        if len(df) < 100:
            return MultiTimeframeAnalysis()
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        current_time = df.index[-1]
        completed_daily_df = df[df.index < current_time.normalize()]
        if len(completed_daily_df) < 1:
            return MultiTimeframeAnalysis()
        daily_df = (
            completed_daily_df.resample("D")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        weekly_df = (
            completed_daily_df.resample("W")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        daily_bias = MarketBias.NEUTRAL
        weekly_bias = MarketBias.NEUTRAL
        daily_support = None
        daily_resistance = None
        weekly_support = None
        weekly_resistance = None
        if len(daily_df) >= 3:
            last_daily_close = daily_df["close"].iloc[-1]
            prev_daily_close = daily_df["close"].iloc[-2]
            if last_daily_close > prev_daily_close:
                daily_bias = MarketBias.BULLISH
            elif last_daily_close < prev_daily_close:
                daily_bias = MarketBias.BEARISH
            daily_support = daily_df["low"].tail(10).min()
            daily_resistance = daily_df["high"].tail(10).max()
        if len(weekly_df) >= 3:
            last_weekly_close = weekly_df["close"].iloc[-1]
            prev_weekly_close = weekly_df["close"].iloc[-2]
            if last_weekly_close > prev_weekly_close:
                weekly_bias = MarketBias.BULLISH
            elif last_weekly_close < prev_weekly_close:
                weekly_bias = MarketBias.BEARISH
            weekly_support = weekly_df["low"].tail(5).min()
            weekly_resistance = weekly_df["high"].tail(5).max()
        volume_profile = calculate_volume_profile(df, window=200)
        return MultiTimeframeAnalysis(
            daily_bias=daily_bias,
            weekly_bias=weekly_bias,
            daily_support=daily_support,
            daily_resistance=daily_resistance,
            weekly_support=weekly_support,
            weekly_resistance=weekly_resistance,
            volume_profile=volume_profile,
        )

    def calculate_confluence_score(
        self,
        current_price: float,
        order_blocks: List[OrderBlock],
        fvgs: List[FairValueGap],
        liquidity: List[LiquidityPool],
        zones: Dict[str, Tuple[float, float]],
        liquidity_sweeps: Dict[str, Any],
        multi_tf: MultiTimeframeAnalysis,
    ) -> float:
        score = 0.0
        factors = 1.0
        (premium_bottom, premium_top) = zones["premium"]
        (discount_bottom, discount_top) = zones["discount"]
        if liquidity_sweeps["bullish_sweep"]:
            score += 4.0
            factors += 4.0
        if liquidity_sweeps["bearish_sweep"]:
            score -= 4.0
            factors += 4.0
        if current_price <= discount_top:
            score += 2.0 if self.swing_bias == MarketBias.BULLISH else -2.0
            factors += 2.0
        elif current_price >= premium_bottom:
            score += 2.0 if self.swing_bias == MarketBias.BEARISH else -2.0
            factors += 2.0
        else:
            score += 0.5
            factors += 0.5
        active_obs = [
            ob for ob in order_blocks if not ob.mitigated and ob.strength >= 0.7
        ]
        for ob in active_obs[:3]:
            ob_bottom = ob.bottom
            ob_top = ob.top
            is_near_bull_ob = (
                ob.type == OrderBlockType.BULLISH
                and ob_bottom <= current_price <= ob_top * 1.005
            )
            is_near_bear_ob = (
                ob.type == OrderBlockType.BEARISH
                and ob_top >= current_price >= ob_bottom * 0.995
            )
            if is_near_bull_ob:
                score += 3.0
                factors += 3.0
            elif is_near_bear_ob:
                score -= 3.0
                factors += 3.0
        active_fvgs = [
            fvg
            for fvg in fvgs
            if not fvg.filled and fvg.top - fvg.bottom > 0.005 * current_price
        ]
        for fvg in active_fvgs[:2]:
            if fvg.type == OrderBlockType.BULLISH and current_price < fvg.top:
                score += 1.5
                factors += 1.5
            elif fvg.type == OrderBlockType.BEARISH and current_price > fvg.bottom:
                score -= 1.5
                factors += 1.5
        active_pools = [lp for lp in liquidity if not lp.swept and lp.strength >= 2]
        for lp in active_pools[:3]:
            rel_dist = abs(current_price - lp.level) / current_price
            if rel_dist < 0.015:
                score += lp.strength * 1.5 if lp.type == "EQL" else -lp.strength * 1.5
                factors += lp.strength * 1.5
        if multi_tf.daily_bias.name == MarketBias.BULLISH.name:
            score += 1.0
            factors += 1.0
        elif multi_tf.daily_bias.name == MarketBias.BEARISH.name:
            score -= 1.0
            factors += 1.0
        if multi_tf.weekly_bias.name == MarketBias.BULLISH.name:
            score += 2.0
            factors += 2.0
        elif multi_tf.weekly_bias.name == MarketBias.BEARISH.name:
            score -= 2.0
            factors += 2.0
        if multi_tf.daily_support and current_price <= multi_tf.daily_support * 1.01:
            score += 1.5
            factors += 1.5
        if (
            multi_tf.daily_resistance
            and current_price >= multi_tf.daily_resistance * 0.99
        ):
            score -= 1.5
            factors += 1.5
        if multi_tf.weekly_support and current_price <= multi_tf.weekly_support * 1.02:
            score += 2.0
            factors += 2.0
        if (
            multi_tf.weekly_resistance
            and current_price >= multi_tf.weekly_resistance * 0.98
        ):
            score -= 2.0
            factors += 2.0
        final_score = score / factors
        return max(-1.0, min(1.0, final_score))

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        current_time_ts = int(df.index[-1].timestamp())
        if current_time_ts == self.last_analyzed_time:
            return self._last_result if self._last_result else {}
        (new_highs, new_lows) = self.find_pivots_luxalgo(df, size=5, internal=False)
        if new_highs:
            if (
                not self.confirmed_swing_highs
                or new_highs[-1].time > self.confirmed_swing_highs[-1].time
            ):
                self.confirmed_swing_highs.append(new_highs[-1])
                if len(self.confirmed_swing_highs) > 20:
                    self.confirmed_swing_highs.pop(0)
        if new_lows:
            if (
                not self.confirmed_swing_lows
                or new_lows[-1].time > self.confirmed_swing_lows[-1].time
            ):
                self.confirmed_swing_lows.append(new_lows[-1])
                if len(self.confirmed_swing_lows) > 20:
                    self.confirmed_swing_lows.pop(0)
        current_price = df["close"].iloc[-1]
        last_sh = self.confirmed_swing_highs[-1] if self.confirmed_swing_highs else None
        last_sl = self.confirmed_swing_lows[-1] if self.confirmed_swing_lows else None
        structures = []
        new_obs = []
        if last_sh and (not last_sh.crossed) and (current_price > last_sh.level):
            self.swing_bias = MarketBias.BULLISH
            last_sh.crossed = True
            structures.append(
                Structure(
                    type=StructureType.BOS,
                    bias=MarketBias.BULLISH,
                    level=last_sh.level,
                    time=int(df.index[-1].timestamp()),
                    pivot=last_sh,
                    is_internal=False,
                )
            )
            ob = self._create_order_block(
                df, last_sh, len(df) - 1, MarketBias.BULLISH, internal=False
            )
            if ob:
                new_obs.append(ob)
        if last_sl and (not last_sl.crossed) and (current_price < last_sl.level):
            self.swing_bias = MarketBias.BEARISH
            last_sl.crossed = True
            structures.append(
                Structure(
                    type=StructureType.BOS,
                    bias=MarketBias.BEARISH,
                    level=last_sl.level,
                    time=int(df.index[-1].timestamp()),
                    pivot=last_sl,
                    is_internal=False,
                )
            )
            ob = self._create_order_block(
                df, last_sl, len(df) - 1, MarketBias.BEARISH, internal=False
            )
            if ob:
                new_obs.append(ob)
        self.last_analyzed_time = current_time_ts
        fvgs = self.find_fvgs(df)
        zones = self.get_premium_discount_zones(df)
        liquidity_pools = self.find_liquidity_pools(df)
        multi_tf = self.get_multi_timeframe_analysis(df)
        last_swing_high_level = last_sh.level if last_sh else None
        last_swing_low_level = last_sl.level if last_sl else None
        liquidity_sweeps = self.detect_liquidity_sweeps(
            df,
            {
                "current_price": current_price,
                "last_swing_high": last_swing_high_level,
                "last_swing_low": last_swing_low_level,
            },
        )
        confluence_score = self.calculate_confluence_score(
            current_price,
            new_obs,
            fvgs,
            liquidity_pools,
            zones,
            liquidity_sweeps,
            multi_tf,
        )
        self._last_result = {
            "current_price": float(current_price),
            "swing_bias": self.swing_bias.name,
            "swing_highs": self.confirmed_swing_highs,
            "swing_lows": self.confirmed_swing_lows,
            "new_structures": structures,
            "new_order_blocks": new_obs,
            "fair_value_gaps": fvgs,
            "zones": zones,
            "liquidity_pools": [
                {
                    "level": lp.level,
                    "type": lp.type,
                    "strength": lp.strength,
                    "swept": lp.swept,
                }
                for lp in liquidity_pools
            ],
            "trailing_extremes": {
                "top": self.trailing.top,
                "bottom": self.trailing.bottom,
            },
            "multi_timeframe": {
                "daily_bias": multi_tf.daily_bias.name,
                "weekly_bias": multi_tf.weekly_bias.name,
                "daily_support": multi_tf.daily_support,
                "daily_resistance": multi_tf.daily_resistance,
                "weekly_support": multi_tf.weekly_support,
                "weekly_resistance": multi_tf.weekly_resistance,
            },
            "liquidity_sweeps": liquidity_sweeps,
            "confluence_score": confluence_score,
            "last_swing_high": last_swing_high_level,
            "last_swing_low": last_swing_low_level,
            "swing_order_blocks": [],
            "internal_order_blocks": [],
        }
        return self._last_result


class EnhancedAlgoSMCStrategy:
    def __init__(self):
        self.min_confidence_threshold = 0.35
        self.min_rr_ratio = 1.2
        self.max_trades_per_day = 5
        self.position_size_risk = 0.015
        self.time_filters = {
            "avoid_low_volume": [23, 0, 1, 2],
            "avoid_market_open": [9, 10],
        }
        self.leverage = 2

    def evaluate_setup(
        self, smc_data: Dict[str, Any], df: pd.DataFrame, balance: float = 10000.0
    ) -> Dict[str, Any]:
        price = float(smc_data["current_price"])
        swing_bias = smc_data["swing_bias"]
        decision = "HOLD"
        sl = 0.0
        tp = 0.0
        entry_reason = ""
        position_size = 0.0
        rr = 0.0
        confidence = float(abs(smc_data.get("confluence_score", 0.0)))
        last_swing_high = smc_data.get("last_swing_high")
        last_swing_low = smc_data.get("last_swing_low")
        try:
            atr_series = compute_atr(df, 14)
            atr = (
                float(atr_series.iloc[-1])
                if len(atr_series) > 0
                else max(0.001 * price, 0.01)
            )
        except Exception:
            atr = max(0.001 * price, 0.01)
        STOP_ATR_MULTIPLIER = 1.5
        obs = []
        obs.extend(smc_data.get("swing_order_blocks", []) or [])
        obs.extend(smc_data.get("internal_order_blocks", []) or [])
        active_obs = [ob for ob in obs if not ob.get("mitigated", False)]
        leverage = getattr(self, "leverage", 2)
        if swing_bias == "BULLISH" and active_obs:
            relevant_obs = [
                ob for ob in active_obs if ob.get("type", "").lower() == "bullish"
            ]
            for ob in relevant_obs:
                ob_bottom = float(ob.get("bottom", 0))
                ob_top = float(ob.get("top", 0))
                if ob_bottom <= price <= ob_top * 1.002:
                    decision = "BUY"
                    sl = ob_bottom - atr * STOP_ATR_MULTIPLIER
                    if last_swing_high and last_swing_high > price:
                        tp = float(last_swing_high)
                    else:
                        tp = price + (price - sl) * 2.5
                    entry_reason = "Mitigation of Bullish OB"
                    risk_amt = balance * getattr(self, "position_size_risk", 0.015)
                    dist = abs(price - sl)
                    if dist > 0:
                        position_size = risk_amt / dist
                    confidence = max(confidence, 0.5)
                    break
        elif swing_bias == "BEARISH" and active_obs:
            relevant_obs = [
                ob for ob in active_obs if ob.get("type", "").lower() == "bearish"
            ]
            for ob in relevant_obs:
                ob_bottom = float(ob.get("bottom", 0))
                ob_top = float(ob.get("top", 0))
                if ob_bottom * 0.998 <= price <= ob_top:
                    decision = "SELL"
                    sl = ob_top + atr * STOP_ATR_MULTIPLIER
                    if last_swing_low and last_swing_low < price:
                        tp = float(last_swing_low)
                    else:
                        tp = price - (sl - price) * 2.5
                    entry_reason = "Mitigation of Bearish OB"
                    risk_amt = balance * getattr(self, "position_size_risk", 0.015)
                    dist = abs(price - sl)
                    if dist > 0:
                        position_size = risk_amt / dist
                    confidence = max(confidence, 0.5)
                    break
        if decision == "HOLD":
            return self._default_hold_response("No OB touch or trend mismatch")
        if decision == "BUY":
            if sl >= price:
                sl = price - 1.5 * atr
            if tp <= price:
                tp = price + 2.5 * atr
        else:
            if sl <= price:
                sl = price + 1.5 * atr
            if tp >= price:
                tp = price - 2.5 * atr
        stop_distance = abs(price - sl)
        reward_distance = abs(tp - price)
        rr = reward_distance / stop_distance if stop_distance > 0 else 0.0
        max_pos_value = balance * leverage * 0.95
        current_pos_value = position_size * price
        if current_pos_value > max_pos_value:
            position_size = max_pos_value / price
        return {
            "decision": decision,
            "leverage": int(leverage),
            "position_size": float(position_size),
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "entry_reason": entry_reason,
            "risk_reward": float(rr),
            "confidence": float(confidence),
            "setup_quality": "OB",
            "atr_value": float(atr),
            "liquidity_sweep": smc_data.get("liquidity_sweeps", {}).get("valid", False),
        }

    def _default_hold_response(self, reason: str) -> Dict[str, Any]:
        return {
            "decision": "HOLD",
            "leverage": 1,
            "position_size": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "entry_reason": reason,
            "risk_reward": 0.0,
            "setup_quality": "LOW",
            "atr_value": 0.0,
            "swing_bias": "NEUTRAL",
            "internal_bias": "NEUTRAL",
            "liquidity_sweep": False,
            "volume_ratio": 0.0,
            "confidence": 0.0,
        }

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        leverage: int,
        risk_per_trade: float = 0.015,
    ) -> float:
        if stop_loss <= 0 or entry_price <= 0:
            return 0.0
        risk_per_trade = min(risk_per_trade, 0.05)
        risk_amount = balance * risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < 0.0001 * entry_price:
            return 0.0
        position_size = risk_amount / stop_distance
        max_position_value = balance * leverage
        max_position_size = max_position_value / entry_price
        final_size = min(position_size, max_position_size)
        return max(0.0, final_size)


class BacktestEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.commission_rate = float(config.get("commission_rate", 0.0004))
        self.maker_commission = float(config.get("maker_commission", 0.0002))
        self.taker_commission = float(config.get("taker_commission", 0.0004))
        self.slippage_pct = float(config.get("slippage_pct", 0.0005))
        self.slippage_model = config.get("slippage_model", "volume_based")
        self.max_positions = int(config.get("max_positions", 1))
        self.leverage = int(config.get("leverage", 3))
        self.use_futures = config.get("use_futures", True)
        self.funding_rate = float(config.get("funding_rate", 0.0001))
        self.risk_per_trade = float(config.get("risk_per_trade", 0.02))
        self.max_correlation_risk = float(config.get("max_correlation_risk", 0.3))
        self.bin_client = BinanceDataClient(use_futures=self.use_futures)
        self.smc_analyzer = SmartMoneyConceptsAnalyzer(
            lookback=config.get("lookback", 50),
            internal_confluence_filter=config.get("internal_confluence_filter", True),
            volatility_method=config.get("volatility_method", "ATR"),
            atr_length=config.get("atr_length", 100),
            ob_mitigation=config.get("ob_mitigation", "HIGHLOW"),
            min_volume_threshold=config.get("min_volume_threshold", 100000.0),
        )
        self.strategy = EnhancedAlgoSMCStrategy()
        self.market_impact_model = config.get("market_impact_model", "square_root")

    def _calculate_market_impact(
        self, position_size: float, avg_volume: float, volatility: float, price: float
    ) -> float:
        if avg_volume <= 0 or position_size <= 0:
            return 0.0
        volume_ratio = position_size / avg_volume
        if self.market_impact_model == "linear":
            base_impact = 0.001 * volume_ratio
        elif self.market_impact_model == "square_root":
            base_impact = 0.002 * np.sqrt(volume_ratio)
        elif self.market_impact_model == "logarithmic":
            base_impact = 0.0005 * np.log(1 + volume_ratio * 100)
        else:
            base_impact = 0.001 * volume_ratio
        volatility_factor = 1 + volatility / price * 10
        return base_impact * volatility_factor

    def _calculate_slippage(
        self,
        price: float,
        position_size: float,
        df: pd.DataFrame,
        is_entry: bool = True,
        decision: str = "BUY",
    ) -> Tuple[float, float]:
        if position_size <= 0 or price <= 0:
            return (0.0, 0.0)
        atr = compute_atr(df.tail(20), 14).iloc[-1] if len(df) > 20 else 0.01 * price
        volatility = atr / price
        avg_volume = (
            df["volume"].tail(20).mean() if len(df) > 20 else position_size * 100
        )
        market_impact = self._calculate_market_impact(
            position_size, avg_volume, volatility, price
        )
        direction_factor = 1.0
        order_type_factor = 1.0
        if is_entry:
            order_type_factor = 1.5
            if decision == "BUY":
                direction_factor = 1.0
            else:
                direction_factor = -1.0
        else:
            order_type_factor = 1.0
            if decision == "BUY":
                direction_factor = -1.0
            else:
                direction_factor = 1.0
        base_slippage_pct = self.slippage_pct * (1 + volatility * 5)
        total_slippage_pct = base_slippage_pct + market_impact
        total_slippage_pct = total_slippage_pct * order_type_factor
        max_slippage = 0.01 if is_entry else 0.005
        total_slippage_pct = min(total_slippage_pct, max_slippage)
        slippage_amount = price * total_slippage_pct * direction_factor
        slippage_pct = total_slippage_pct * direction_factor
        return (slippage_amount, slippage_pct)

    def _calculate_commission(
        self,
        price: float,
        quantity: float,
        is_maker: bool = False,
        position_type: str = "entry",
    ) -> float:
        if price <= 0 or quantity <= 0:
            return 0.0
        if is_maker:
            commission_rate = self.maker_commission
        else:
            commission_rate = self.taker_commission
        if position_type == "exit":
            commission_rate = commission_rate * 1.1
        commission = abs(price * quantity) * commission_rate
        min_commission = 0.1
        return max(commission, min_commission)

    def _apply_slippage_and_commission(
        self,
        price: float,
        size: float,
        is_entry: bool,
        decision: str = "BUY",
        df: Optional[pd.DataFrame] = None,
        is_maker: bool = False,
        position_type: str = "entry",
    ) -> Tuple[float, float]:
        slippage_pct = self.slippage_pct
        if df is not None and len(df) > 0:
            atr = compute_atr(df).iloc[-1]
            volatility = atr / price
            slippage_pct = self.slippage_pct * (1 + volatility * 10)
        slippage_amount = price * slippage_pct
        adjusted_price = price
        if is_entry:
            if decision == "BUY":
                adjusted_price = price + slippage_amount
            else:
                adjusted_price = price - slippage_amount
        elif decision == "BUY":
            adjusted_price = price - slippage_amount
        else:
            adjusted_price = price + slippage_amount
        commission = adjusted_price * size * self.commission_rate
        return (adjusted_price, commission)

    def _calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        decision: str,
        leverage: int,
        entry_commission: float = 0.0,
        exit_commission: float = 0.0,
        funding_payments: float = 0.0,
        slippage_loss: float = 0.0,
    ) -> float:
        if size <= 0 or leverage <= 0:
            return 0.0
        if decision == "BUY":
            base_pnl = (exit_price - entry_price) * size
        else:
            base_pnl = (entry_price - exit_price) * size
        total_costs = entry_commission + exit_commission + slippage_loss
        final_pnl = base_pnl - total_costs - funding_payments
        logger.debug(
            f"PnL calc: base_pnl={base_pnl:.4f}, total_costs={total_costs:.4f}, funding={funding_payments:.4f}, final_pnl={final_pnl:.4f}"
        )
        return final_pnl

    def _calculate_funding_payments(
        self,
        entry_time: datetime,
        exit_time: datetime,
        position_size: float,
        entry_price: float,
        leverage: int,
        hourly_rate: float = 0.0001,
    ) -> float:
        if not self.use_futures or position_size <= 0:
            return 0.0
        hours_in_position = (exit_time - entry_time).total_seconds() / 3600
        notional_value = position_size * entry_price * leverage
        funding_payment = notional_value * hourly_rate * hours_in_position
        return -funding_payment if hourly_rate > 0 else funding_payment

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        annual_factor = 365 * 24 / len(returns)
        annual_return = (1 + returns.mean()) ** annual_factor - 1
        annual_volatility = returns.std() * np.sqrt(annual_factor)
        if annual_volatility == 0:
            return 0.0
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        return max(-10.0, min(10.0, sharpe))

    def _calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        if len(returns) < 2:
            return 0.0
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0.0
        annual_factor = 365 * 24 / len(returns)
        annual_return = (1 + returns.mean()) * annual_factor - 1
        downside_volatility = negative_returns.std() * np.sqrt(annual_factor)
        if downside_volatility == 0:
            return 0.0
        sortino = (annual_return - risk_free_rate) / downside_volatility
        return max(-10.0, min(10.0, sortino))

    def _calculate_trade_metrics(self, trades: List[TradeSimulation]) -> Dict[str, Any]:
        if not trades:
            return {
                "avg_trade_duration": timedelta(),
                "avg_profit_per_trade": 0.0,
                "avg_loss_per_trade": 0.0,
                "largest_winning_trade": 0.0,
                "largest_losing_trade": 0.0,
                "profit_factor": 0.0,
                "max_mae": 0.0,
                "max_mfe": 0.0,
            }
        durations = [trade.holding_period for trade in trades]
        avg_duration = sum(durations, timedelta()) / len(durations)
        winning_trades = [trade for trade in trades if trade.pnl > 0]
        losing_trades = [trade for trade in trades if trade.pnl <= 0]
        avg_profit = (
            np.mean([trade.pnl for trade in winning_trades]) if winning_trades else 0.0
        )
        avg_loss = (
            np.mean([abs(trade.pnl) for trade in losing_trades])
            if losing_trades
            else 0.0
        )
        largest_winning = max((trade.pnl for trade in trades)) if trades else 0.0
        largest_losing = min((trade.pnl for trade in trades)) if trades else 0.0
        gross_profit = sum((trade.pnl for trade in trades if trade.pnl > 0))
        gross_loss = abs(sum((trade.pnl for trade in trades if trade.pnl <= 0)))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        max_mae = (
            max((trade.max_adverse_excursion for trade in trades)) if trades else 0.0
        )
        max_mfe = (
            max((trade.max_favorable_excursion for trade in trades)) if trades else 0.0
        )
        return {
            "avg_trade_duration": avg_duration,
            "avg_profit_per_trade": avg_profit,
            "avg_loss_per_trade": avg_loss,
            "largest_winning_trade": largest_winning,
            "largest_losing_trade": largest_losing,
            "profit_factor": profit_factor,
            "max_mae": max_mae,
            "max_mfe": max_mfe,
        }

    def plot_backtest_results(self, result: BacktestResult, file_path: Path):
        logger.info(f"Создание графиков результатов: {file_path.parent}")
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "text.color": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.color": "#CCCCCC",
                "grid.linestyle": "--",
                "grid.alpha": 0.7,
                "font.family": "DejaVu Sans",
            }
        )
        bin_client = BinanceDataClient(use_futures=self.use_futures)
        plot_df = bin_client.get_klines(
            symbol=result.symbol,
            interval=result.interval,
            start_time=int(result.start_time.timestamp() * 1000),
            end_time=int(result.end_time.timestamp() * 1000),
            limit=500,
        )
        file_path.parent.mkdir(exist_ok=True)
        self._create_main_chart(result, plot_df, file_path.parent / "chart_main.png")
        self._create_performance_dashboard(
            result, file_path.parent / "chart_performance.png"
        )
        self._create_trade_analysis(result, file_path.parent / "chart_trades.png")
        self._create_combined_chart(result, plot_df, file_path)
        logger.info(f"Графики успешно сохранены в: {file_path.parent}")

    def _create_main_chart(
        self, result: BacktestResult, df: pd.DataFrame, file_path: Path
    ):
        if df.empty:
            logger.warning("Нет данных для основного графика")
            return
        (fig, axes) = plt.subplots(
            2, 1, figsize=(16, 10), height_ratios=[3, 1], gridspec_kw={"hspace": 0.1}
        )
        ax_price = axes[0]
        ax_volume = axes[1]
        self._plot_candles_improved(ax_price, df)
        self._plot_trade_markers(ax_price, result.trades, df)
        self._plot_trade_zones(ax_price, result.trades, df)
        if result.trades and result.trades[-1].smc_context:
            self._plot_smc_levels(ax_price, df, result.trades[-1].smc_context)
        self._plot_volume(ax_volume, df)
        ax_price.set_title(
            f"{result.symbol} | {result.interval} | Return: {result.total_return:.1%} | Trades: {result.total_trades}",
            fontsize=14,
            fontweight="bold",
        )
        ax_price.set_ylabel("Price", fontsize=11)
        ax_price.legend(loc="upper left", fontsize=8, ncol=2)
        ax_price.grid(True, alpha=0.5)
        ax_volume.set_ylabel("Volume", fontsize=11)
        ax_volume.set_xlabel("Date", fontsize=11)
        ax_volume.grid(True, alpha=0.5)
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Основной график сохранён: {file_path}")

    def _plot_candles_improved(self, ax, df: pd.DataFrame):
        if len(df) < 2:
            return
        time_diff = (df.index[1] - df.index[0]).total_seconds() / 86400
        width = time_diff * 0.7
        up_mask = df["close"] >= df["open"]
        down_mask = df["close"] < df["open"]
        up_df = df[up_mask]
        if not up_df.empty:
            ax.bar(
                up_df.index,
                up_df["close"] - up_df["open"],
                bottom=up_df["open"],
                width=width,
                color="#26A69A",
                edgecolor="#1B7D71",
                linewidth=0.5,
            )
            ax.vlines(
                up_df.index, up_df["low"], up_df["high"], color="#1B7D71", linewidth=0.8
            )
        down_df = df[down_mask]
        if not down_df.empty:
            ax.bar(
                down_df.index,
                down_df["open"] - down_df["close"],
                bottom=down_df["close"],
                width=width,
                color="#EF5350",
                edgecolor="#C62828",
                linewidth=0.5,
            )
            ax.vlines(
                down_df.index,
                down_df["low"],
                down_df["high"],
                color="#C62828",
                linewidth=0.8,
            )

    def _plot_trade_markers(self, ax, trades: List[TradeSimulation], df: pd.DataFrame):
        if not trades:
            return
        buy_entries = [
            (t.entry_time, t.entry_price) for t in trades if t.decision == "BUY"
        ]
        sell_entries = [
            (t.entry_time, t.entry_price) for t in trades if t.decision == "SELL"
        ]
        profitable_exits = [(t.exit_time, t.exit_price) for t in trades if t.pnl > 0]
        losing_exits = [(t.exit_time, t.exit_price) for t in trades if t.pnl <= 0]
        if buy_entries:
            (times, prices) = zip(*buy_entries)
            ax.scatter(
                times,
                prices,
                marker="^",
                s=120,
                c="#2196F3",
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label="BUY Entry",
            )
        if sell_entries:
            (times, prices) = zip(*sell_entries)
            ax.scatter(
                times,
                prices,
                marker="v",
                s=120,
                c="#FF9800",
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label="SELL Entry",
            )
        if profitable_exits:
            (times, prices) = zip(*profitable_exits)
            ax.scatter(
                times,
                prices,
                marker="o",
                s=100,
                c="#4CAF50",
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label="Profit Exit",
            )
        if losing_exits:
            (times, prices) = zip(*losing_exits)
            ax.scatter(
                times,
                prices,
                marker="x",
                s=100,
                c="#F44336",
                linewidth=2,
                zorder=10,
                label="Loss Exit",
            )
        recent_trades = trades[-10:] if len(trades) > 10 else trades
        for trade in recent_trades:
            color = "#4CAF50" if trade.pnl > 0 else "#F44336"
            ax.plot(
                [trade.entry_time, trade.exit_time],
                [trade.entry_price, trade.exit_price],
                color=color,
                linestyle="--",
                alpha=0.4,
                linewidth=1,
            )

    def _plot_trade_zones(self, ax, trades: List[TradeSimulation], df: pd.DataFrame):
        if not trades:
            return
        recent_trades = trades[-5:] if len(trades) > 5 else trades
        for i, trade in enumerate(recent_trades):
            alpha = 0.15 + i * 0.05
            entry_time = trade.entry_time
            exit_time = trade.exit_time
            time_buffer = (exit_time - entry_time) * 0.1
            start_time = (
                entry_time - time_buffer
                if isinstance(time_buffer, timedelta)
                else entry_time
            )
            end_time = (
                exit_time + time_buffer
                if isinstance(time_buffer, timedelta)
                else exit_time
            )
            if trade.decision == "BUY":
                ax.axhspan(
                    trade.stop_loss,
                    trade.entry_price,
                    xmin=0,
                    xmax=1,
                    alpha=0.05,
                    color="red",
                )
                ax.hlines(
                    trade.stop_loss,
                    start_time,
                    end_time,
                    colors="#F44336",
                    linestyles=":",
                    linewidth=1.5,
                    alpha=0.6,
                )
                ax.hlines(
                    trade.take_profit,
                    start_time,
                    end_time,
                    colors="#4CAF50",
                    linestyles=":",
                    linewidth=1.5,
                    alpha=0.6,
                )
            else:
                ax.axhspan(
                    trade.entry_price,
                    trade.stop_loss,
                    xmin=0,
                    xmax=1,
                    alpha=0.05,
                    color="red",
                )
                ax.hlines(
                    trade.stop_loss,
                    start_time,
                    end_time,
                    colors="#F44336",
                    linestyles=":",
                    linewidth=1.5,
                    alpha=0.6,
                )
                ax.hlines(
                    trade.take_profit,
                    start_time,
                    end_time,
                    colors="#4CAF50",
                    linestyles=":",
                    linewidth=1.5,
                    alpha=0.6,
                )

    def _plot_volume(self, ax, df: pd.DataFrame):
        if len(df) < 2:
            return
        time_diff = (df.index[1] - df.index[0]).total_seconds() / 86400
        width = time_diff * 0.7
        colors = [
            "#26A69A" if c >= o else "#EF5350"
            for (c, o) in zip(df["close"], df["open"])
        ]
        ax.bar(df.index, df["volume"], width=width, color=colors, alpha=0.7)
        avg_vol = df["volume"].rolling(20).mean()
        ax.plot(
            df.index,
            avg_vol,
            color="#FF9800",
            linewidth=1.5,
            label="Avg Volume (20)",
            alpha=0.8,
        )
        ax.legend(loc="upper left", fontsize=8)

    def _create_performance_dashboard(self, result: BacktestResult, file_path: Path):
        (fig, axes) = plt.subplots(2, 2, figsize=(14, 10))
        self._plot_equity_improved(axes[0, 0], result)
        self._plot_drawdown_improved(axes[0, 1], result)
        self._plot_pnl_improved(axes[1, 0], result)
        self._plot_stats_table(axes[1, 1], result)
        fig.suptitle(
            f"Performance Dashboard: {result.symbol} | {result.interval}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Дашборд производительности сохранён: {file_path}")

    def _plot_equity_improved(self, ax, result: BacktestResult):
        if result.equity_curve is None or result.equity_curve.empty:
            ax.text(
                0.5,
                0.5,
                "No equity data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Equity Curve")
            return
        equity = result.equity_curve
        ax.plot(
            equity.index, equity.values, color="#1976D2", linewidth=2, label="Equity"
        )
        ax.axhline(
            result.initial_balance,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Initial: ${result.initial_balance:,.0f}",
        )
        running_max = equity.cummax()
        ax.plot(
            running_max.index,
            running_max.values,
            color="#4CAF50",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Peak",
        )
        ax.fill_between(
            equity.index,
            equity.values,
            running_max.values,
            where=equity.values < running_max.values,
            color="#FFCDD2",
            alpha=0.5,
        )
        final_balance = equity.iloc[-1]
        ax.annotate(
            f"${final_balance:,.0f}",
            xy=(equity.index[-1], final_balance),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="#1976D2",
        )
        ax.set_title(
            f"Equity Curve | Final: ${result.final_balance:,.0f} ({result.total_return:+.1%})",
            fontsize=11,
        )
        ax.set_ylabel("Balance ($)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    def _plot_drawdown_improved(self, ax, result: BacktestResult):
        if result.drawdown_curve is None or result.drawdown_curve.empty:
            ax.text(
                0.5,
                0.5,
                "No drawdown data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Drawdown")
            return
        dd = result.drawdown_curve * 100
        ax.fill_between(dd.index, 0, dd.values, color="#FFCDD2", alpha=0.7)
        ax.plot(dd.index, dd.values, color="#D32F2F", linewidth=1.5)
        max_dd = dd.max()
        max_dd_idx = dd.idxmax()
        if not (math.isnan(max_dd) or math.isinf(max_dd)):
            ax.axhline(
                max_dd, color="#B71C1C", linestyle="--", linewidth=1.5, alpha=0.7
            )
            ax.scatter(
                [max_dd_idx], [max_dd], color="#B71C1C", s=100, zorder=5, marker="v"
            )
            ax.annotate(
                f"Max: {max_dd:.1f}%",
                xy=(max_dd_idx, max_dd),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                color="#B71C1C",
            )
        ax.set_title(f"Drawdown | Max: {result.max_drawdown:.1%}", fontsize=11)
        ax.set_ylabel("Drawdown (%)")
        ax.set_ylim(bottom=0)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.5)

    def _plot_pnl_improved(self, ax, result: BacktestResult):
        if not result.trades:
            ax.text(
                0.5,
                0.5,
                "No trades",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("PnL Distribution")
            return
        pnls = [t.pnl for t in result.trades]
        profits = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        all_pnls = pnls
        if len(set(all_pnls)) > 1:
            bins = np.linspace(min(all_pnls), max(all_pnls), 20)
        else:
            bins = 10
        ax.hist(
            profits,
            bins=bins,
            color="#4CAF50",
            alpha=0.7,
            label=f"Profits ({len(profits)})",
            edgecolor="white",
        )
        ax.hist(
            losses,
            bins=bins,
            color="#F44336",
            alpha=0.7,
            label=f"Losses ({len(losses)})",
            edgecolor="white",
        )
        avg_pnl = np.mean(pnls)
        ax.axvline(
            avg_pnl,
            color="#1976D2",
            linestyle="--",
            linewidth=2,
            label=f"Avg: ${avg_pnl:.2f}",
        )
        ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax.set_title(f"PnL Distribution | Avg: ${avg_pnl:.2f}", fontsize=11)
        ax.set_xlabel("PnL ($)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.5)

    def _plot_stats_table(self, ax, result: BacktestResult):
        ax.axis("off")
        trade_metrics = self._calculate_trade_metrics(result.trades)
        stats_text = f"\nPERFORMANCE SUMMARY\n{'=' * 40}\nInitial Balance:     ${result.initial_balance:>12,.0f}\nFinal Balance:       ${result.final_balance:>12,.0f}\nTotal Return:        {result.total_return:>12.2%}\nAnnualized Return:   {result.annualized_return:>12.2%}\n\nTRADING STATISTICS\n{'=' * 40}\nTotal Trades:        {result.total_trades:>12}\nWinning Trades:      {result.winning_trades:>12}\nWin Rate:            {result.win_rate:>12.1%}\nProfit Factor:       {result.profit_factor:>12.2f}\n\nRISK METRICS\n{'=' * 40}\nMax Drawdown:        {result.max_drawdown:>12.2%}\nSharpe Ratio:        {result.sharpe_ratio:>12.2f}\nSortino Ratio:       {result.sortino_ratio:>12.2f}\nRecovery Factor:     {result.recovery_factor:>12.2f}\n\nTRADE ANALYSIS\n{'=' * 40}\nAvg Profit:          ${trade_metrics['avg_profit_per_trade']:>12,.2f}\nAvg Loss:            ${trade_metrics['avg_loss_per_trade']:>12,.2f}\nBest Trade:          ${trade_metrics['largest_winning_trade']:>12,.2f}\nWorst Trade:         ${trade_metrics['largest_losing_trade']:>12,.2f}\n".strip()
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#F5F5F5",
                edgecolor="#CCCCCC",
                alpha=0.9,
            ),
        )

    def _create_trade_analysis(self, result: BacktestResult, file_path: Path):
        if not result.trades:
            logger.info("Нет сделок для анализа")
            return
        (fig, axes) = plt.subplots(2, 2, figsize=(14, 10))
        self._plot_cumulative_pnl(axes[0, 0], result.trades)
        self._plot_monthly_performance(axes[0, 1], result.trades)
        self._plot_holding_vs_pnl(axes[1, 0], result.trades)
        self._plot_rr_analysis(axes[1, 1], result.trades)
        fig.suptitle(
            f"Trade Analysis: {result.symbol} | {len(result.trades)} trades",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Анализ сделок сохранён: {file_path}")

    def _plot_cumulative_pnl(self, ax, trades: List[TradeSimulation]):
        if not trades:
            return
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        times = [t.exit_time for t in sorted_trades]
        cum_pnl = np.cumsum([t.pnl for t in sorted_trades])
        colors = ["#4CAF50" if pnl >= 0 else "#F44336" for pnl in cum_pnl]
        ax.plot(times, cum_pnl, color="#1976D2", linewidth=2)
        ax.fill_between(
            times, 0, cum_pnl, where=np.array(cum_pnl) >= 0, color="#C8E6C9", alpha=0.5
        )
        ax.fill_between(
            times, 0, cum_pnl, where=np.array(cum_pnl) < 0, color="#FFCDD2", alpha=0.5
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_title("Cumulative PnL", fontsize=11)
        ax.set_ylabel("Cumulative PnL ($)")
        ax.grid(True, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    def _plot_monthly_performance(self, ax, trades: List[TradeSimulation]):
        if not trades:
            return
        monthly_data = {}
        for trade in trades:
            month = trade.entry_time.strftime("%Y-%m")
            if month not in monthly_data:
                monthly_data[month] = {"pnl": 0, "wins": 0, "total": 0}
            monthly_data[month]["pnl"] += trade.pnl
            monthly_data[month]["total"] += 1
            if trade.pnl > 0:
                monthly_data[month]["wins"] += 1
        months = sorted(monthly_data.keys())
        pnls = [monthly_data[m]["pnl"] for m in months]
        win_rates = [
            monthly_data[m]["wins"] / monthly_data[m]["total"] * 100
            if monthly_data[m]["total"] > 0
            else 0
            for m in months
        ]
        x = np.arange(len(months))
        width = 0.4
        colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
        bars = ax.bar(x, pnls, width, color=colors, alpha=0.7, label="PnL")
        ax2 = ax.twinx()
        ax2.plot(
            x,
            win_rates,
            "o-",
            color="#FF9800",
            linewidth=2,
            markersize=8,
            label="Win Rate",
        )
        ax2.set_ylabel("Win Rate (%)", color="#FF9800")
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis="y", labelcolor="#FF9800")
        ax.set_title("Monthly Performance", fontsize=11)
        ax.set_ylabel("PnL ($)")
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.5)
        (lines1, labels1) = ax.get_legend_handles_labels()
        (lines2, labels2) = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    def _plot_holding_vs_pnl(self, ax, trades: List[TradeSimulation]):
        if not trades:
            return
        holding_hours = [t.holding_period.total_seconds() / 3600 for t in trades]
        pnls = [t.pnl for t in trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        sizes = (
            [abs(p) / max(abs(min(pnls)), abs(max(pnls))) * 200 + 50 for p in pnls]
            if pnls
            else [100]
        )
        ax.scatter(
            holding_hours, pnls, c=colors, s=sizes, alpha=0.6, edgecolors="black"
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        if len(holding_hours) > 2:
            z = np.polyfit(holding_hours, pnls, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(holding_hours), max(holding_hours), 100)
            ax.plot(x_line, p(x_line), "--", color="#1976D2", alpha=0.7, label=f"Trend")
        ax.set_title("Holding Time vs PnL", fontsize=11)
        ax.set_xlabel("Holding Time (hours)")
        ax.set_ylabel("PnL ($)")
        ax.grid(True, alpha=0.5)

    def _plot_rr_analysis(self, ax, trades: List[TradeSimulation]):
        if not trades:
            return
        expected_rr = [t.smc_context.get("risk_reward", 0) for t in trades]
        realized_rr = [t.risk_reward_ratio for t in trades]
        x = np.arange(len(trades))
        width = 0.35
        ax.bar(
            x - width / 2,
            expected_rr,
            width,
            label="Expected R:R",
            color="#2196F3",
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            realized_rr,
            width,
            label="Realized R:R",
            color="#4CAF50" if sum(realized_rr) > 0 else "#F44336",
            alpha=0.7,
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.axhline(
            1,
            color="#FF9800",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Break-even",
        )
        ax.set_title("Risk/Reward Analysis", fontsize=11)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("R:R Ratio")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.5)
        if len(trades) > 20:
            step = len(trades) // 10
            ax.set_xticks(x[::step])

    def _create_combined_chart(
        self, result: BacktestResult, df: pd.DataFrame, file_path: Path
    ):
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.25, wspace=0.2)
        ax_main = fig.add_subplot(gs[0, :])
        if not df.empty:
            self._plot_candles_improved(ax_main, df)
            self._plot_trade_markers(ax_main, result.trades, df)
        ax_main.set_title(
            f"{result.symbol} | {result.interval} | Total Return: {result.total_return:.2%}",
            fontsize=13,
            fontweight="bold",
        )
        ax_main.set_ylabel("Price")
        ax_main.legend(loc="upper left", fontsize=8)
        ax_main.grid(True, alpha=0.5)
        ax_equity = fig.add_subplot(gs[1, 0])
        self._plot_equity_improved(ax_equity, result)
        ax_dd = fig.add_subplot(gs[1, 1])
        self._plot_drawdown_improved(ax_dd, result)
        ax_pnl = fig.add_subplot(gs[2, 0])
        self._plot_pnl_improved(ax_pnl, result)
        ax_stats = fig.add_subplot(gs[2, 1])
        self._plot_stats_table(ax_stats, result)
        plt.savefig(file_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Комбинированный график сохранён: {file_path}")

    def _plot_smc_levels(self, ax, df, smc_context):
        try:
            zones = smc_context.get("zones", {})
            labels_added = set()
            if zones:
                if "premium" in zones:
                    zone = zones["premium"]
                    if isinstance(zone, (tuple, list)) and len(zone) == 2:
                        (bottom, top) = (float(zone[0]), float(zone[1]))
                        if self._is_valid_level(bottom) and self._is_valid_level(top):
                            ax.axhspan(
                                bottom,
                                top,
                                alpha=0.15,
                                color="#FFCDD2",
                                label="Premium Zone"
                                if "Premium" not in labels_added
                                else "",
                            )
                            labels_added.add("Premium")
                if "discount" in zones:
                    zone = zones["discount"]
                    if isinstance(zone, (tuple, list)) and len(zone) == 2:
                        (bottom, top) = (float(zone[0]), float(zone[1]))
                        if self._is_valid_level(bottom) and self._is_valid_level(top):
                            ax.axhspan(
                                bottom,
                                top,
                                alpha=0.15,
                                color="#C8E6C9",
                                label="Discount Zone"
                                if "Discount" not in labels_added
                                else "",
                            )
                            labels_added.add("Discount")
                if "equilibrium" in zones:
                    zone = zones["equilibrium"]
                    if isinstance(zone, (tuple, list)) and len(zone) == 2:
                        (bottom, top) = (float(zone[0]), float(zone[1]))
                        if self._is_valid_level(bottom) and self._is_valid_level(top):
                            ax.axhspan(
                                bottom,
                                top,
                                alpha=0.1,
                                color="#E0E0E0",
                                label="Equilibrium"
                                if "Equilibrium" not in labels_added
                                else "",
                            )
                            labels_added.add("Equilibrium")
            swing_obs = smc_context.get("swing_order_blocks", [])
            for i, ob in enumerate(swing_obs[:3]):
                if not ob.get("mitigated", False):
                    ob_type = ob.get("type", "")
                    bottom = float(ob.get("bottom", 0))
                    top = float(ob.get("top", 0))
                    if (
                        self._is_valid_level(bottom)
                        and self._is_valid_level(top)
                        and (top > bottom)
                    ):
                        color = "#2196F3" if ob_type == "Bullish" else "#9C27B0"
                        alpha = max(0.1, 0.25 - i * 0.05)
                        label_key = f"{('Bull' if ob_type == 'Bullish' else 'Bear')} OB"
                        ax.axhspan(
                            bottom,
                            top,
                            alpha=alpha,
                            color=color,
                            label=label_key if label_key not in labels_added else "",
                        )
                        labels_added.add(label_key)
            liquidity_pools = smc_context.get("liquidity_pools", [])
            for lp in liquidity_pools[:5]:
                if not lp.get("swept", False):
                    level = float(lp.get("level", 0))
                    lp_type = lp.get("type", "")
                    strength = lp.get("strength", 1)
                    if self._is_valid_level(level):
                        color = "#FFC107" if lp_type == "EQH" else "#FF5722"
                        linestyle = "--" if lp_type == "EQH" else ":"
                        label_key = f"{lp_type}"
                        ax.axhline(
                            level,
                            color=color,
                            linestyle=linestyle,
                            alpha=0.6,
                            linewidth=1 + strength * 0.3,
                            label=f"{lp_type} (x{strength})"
                            if label_key not in labels_added
                            else "",
                        )
                        labels_added.add(label_key)
            trailing = smc_context.get("trailing_extremes", {})
            if trailing:
                top_val = trailing.get("top", -float("inf"))
                bottom_val = trailing.get("bottom", float("inf"))
                if self._is_valid_level(top_val):
                    ax.axhline(
                        top_val,
                        color="#00BCD4",
                        linestyle="-",
                        alpha=0.5,
                        linewidth=1.5,
                        label=f"Range High" if "Range High" not in labels_added else "",
                    )
                    labels_added.add("Range High")
                if self._is_valid_level(bottom_val):
                    ax.axhline(
                        bottom_val,
                        color="#E91E63",
                        linestyle="-",
                        alpha=0.5,
                        linewidth=1.5,
                        label=f"Range Low" if "Range Low" not in labels_added else "",
                    )
                    labels_added.add("Range Low")
        except Exception as e:
            logger.warning(f"Ошибка при построении SMC уровней: {e}")

    def _is_valid_level(self, value: float) -> bool:
        if value is None:
            return False
        try:
            v = float(value)
            return not (math.isinf(v) or math.isnan(v) or v <= 0)
        except (TypeError, ValueError):
            return False


class EnhancedBacktestEngine(BacktestEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_volume_threshold = float(config.get("min_volume_threshold", 100000.0))
        self.max_correlation_risk = float(config.get("max_correlation_risk", 0.3))
        self.position_sizing_method = config.get("position_sizing_method", "risk_based")
        self.time_filters = config.get(
            "time_filters",
            {
                "avoid_low_volume": [23, 0, 1, 2],
                "avoid_market_open": [9, 10],
                "prefer_high_liquidity": [14, 15, 16, 17, 18, 19, 20, 21],
            },
        )
        self.session_trades_limit = int(config.get("session_trades_limit", 5))
        self.daily_loss_limit_pct = float(config.get("daily_loss_limit_pct", 0.05))
        self.leverage = int(config.get("leverage", 3))

    def _validate_trade_setup(
        self,
        decision: Dict[str, Any],
        current_price: float,
        smc_data: Dict[str, Any],
        df: pd.DataFrame,
    ) -> bool:
        if decision.get("setup_quality") == "DEBUG":
            logger.debug("✅ Пропускаем валидацию для DEBUG сигнала")
            return True
        min_rr = 1.2
        if decision.get("risk_reward", 0) < min_rr:
            logger.debug(
                f"Rejected: Insufficient R:R ({decision.get('risk_reward', 0):.2f} < {min_rr})"
            )
            return False
        min_confidence = 0.35
        if abs(decision.get("confidence", 0)) < min_confidence:
            logger.debug(
                f"Rejected: Low confidence score ({decision.get('confidence', 0):.2f} < {min_confidence})"
            )
            return False
        current_volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].tail(20).mean()
        if current_volume < avg_volume * 0.7:
            logger.debug(
                f"Rejected: Low volume ({current_volume:,.0f} < {avg_volume * 0.7:,.0f})"
            )
            return False
        current_time = df.index[-1]
        current_hour = current_time.hour
        if current_hour in self.time_filters.get("avoid_low_volume", []):
            logger.debug(f"Rejected: Low volume hour ({current_hour}:00)")
            return False
        position_value = current_price * decision.get("position_size", 0)
        if self.use_futures and position_value < 50:
            logger.debug(
                f"Rejected: Insufficient position value (${position_value:,.2f} < $50)"
            )
            return False
        atr = decision.get("atr_value", 0)
        stop_loss = decision.get("stop_loss", 0)
        if current_price <= 0 or atr <= 0 or stop_loss <= 0:
            logger.debug("Rejected: Invalid price or ATR/SL values.")
            return False
        stop_distance = abs(current_price - stop_loss)
        if stop_distance < 0.3 * atr or stop_distance > 3.0 * atr:
            logger.debug(
                f"Rejected: Unrealistic stop distance ({stop_distance:.4f} vs ATR {atr:.4f})"
            )
            return False
        multi_tf = smc_data.get("multi_timeframe", {})
        daily_bias = multi_tf.get("daily_bias", "NEUTRAL")
        weekly_bias = multi_tf.get("weekly_bias", "NEUTRAL")
        if decision["decision"] == "BUY":
            if daily_bias == "BEARISH" and weekly_bias == "BEARISH":
                logger.debug(
                    "Rejected: Conflicting multi-timeframe bias (daily & weekly BEARISH)"
                )
                return False
        elif decision["decision"] == "SELL":
            if daily_bias == "BULLISH" and weekly_bias == "BULLISH":
                logger.debug(
                    "Rejected: Conflicting multi-timeframe bias (daily & weekly BULLISH)"
                )
                return False
        return True

    def _calculate_dynamic_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: float = 0.015,
        volatility_factor: float = 1.0,
    ) -> float:
        return self.strategy.calculate_position_size(
            balance, entry_price, stop_loss, self.leverage, risk_pct
        )

    def run_enhanced_backtest(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        initial_balance: float = 10000.0,
    ) -> BacktestResult:
        logger.info(
            f"🚀 Запуск УЛУЧШЕННОГО бектеста для {symbol} на интервале {interval}"
        )
        logger.info(f"⏰ Период: {start_time} - {end_time}")
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        try:
            df = self.bin_client.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_ts,
                end_time=end_ts,
                limit=5000,
            )
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
        if df.empty or len(df) < 150:
            raise ValueError(
                f"Недостаточно данных для {symbol} на интервале {interval}. Требуется минимум 150 свечей."
            )
        avg_volume = df["volume"].tail(100).mean()
        logger.info(f"📊 Средний объем за последние 100 свечей: ${avg_volume:,.2f}")
        if avg_volume < self.min_volume_threshold:
            logger.warning(
                f"⚠️ Низкая ликвидность для {symbol}: avg_volume=${avg_volume:,.2f} < ${self.min_volume_threshold:,.2f}"
            )
            if not self.config.get("allow_low_liquidity", False):
                logger.warning("Continuing backtest despite low liquidity.")
        logger.info(f"✅ Получено {len(df)} свечей для анализа")
        balance = initial_balance
        equity = initial_balance
        positions = {}
        trades = []
        equity_curve = []
        timestamps = []
        max_equity = equity
        max_drawdown = 0.0
        total_commission = 0.0
        winning_trades = 0
        total_pnl = 0.0
        total_trade_count = 0
        daily_trades_count = 0
        daily_pnl = 0.0
        last_trade_date = None
        smc_analysis_cache = []
        for j in range(150, len(df)):
            try:
                smc_analysis_cache.append(self.smc_analyzer.analyze(df.iloc[: j + 1]))
            except Exception as e:
                logger.warning(f"Error analyzing data at {df.index[j]}: {e}")
                smc_analysis_cache.append(None)
        active_swing_obs: List[OrderBlock] = []
        for i in range(150, len(df) - 1):
            current_time = df.index[i]
            current_date = current_time.date()
            current_price = df["close"].iloc[i]
            if last_trade_date != current_date:
                daily_trades_count = 0
                daily_pnl = 0.0
                last_trade_date = current_date
            for asset, pos in list(positions.items()):
                if pos["decision"] == "BUY":
                    unrealized_pnl = (
                        (current_price - pos["entry_price"])
                        * pos["position_size"]
                        * pos["leverage"]
                    )
                else:
                    unrealized_pnl = (
                        (pos["entry_price"] - current_price)
                        * pos["position_size"]
                        * pos["leverage"]
                    )
                pos["unrealized_pnl"] = unrealized_pnl
            equity = balance + sum(
                (pos.get("unrealized_pnl", 0) for pos in positions.values())
            )
            smc_data = (
                smc_analysis_cache[i - 150]
                if i - 150 < len(smc_analysis_cache)
                else None
            )
            if smc_data is None:
                continue
            if smc_data.get("new_order_blocks"):
                for new_ob in smc_data["new_order_blocks"]:
                    active_swing_obs.insert(0, new_ob)
                if len(active_swing_obs) > 20:
                    active_swing_obs = active_swing_obs[:20]
            active_swing_obs = [
                ob
                for ob in active_swing_obs
                if not (
                    ob.type == OrderBlockType.BEARISH
                    and current_price > ob.top
                    or (ob.type == OrderBlockType.BULLISH and current_price < ob.bottom)
                )
            ]
            smc_data["swing_order_blocks"] = [
                {
                    "type": ob.type.value,
                    "top": ob.top,
                    "bottom": ob.bottom,
                    "mitigated": ob.mitigated,
                }
                for ob in active_swing_obs
            ]
            current_df = df.iloc[: i + 1]
            try:
                decision = self.strategy.evaluate_setup(smc_data, current_df, balance)
            except Exception as e:
                logger.warning(f"Error evaluating setup at {current_time}: {e}")
                decision = self.strategy._default_hold_response(str(e))
            if i % 50 == 0:
                logger.info(
                    f"[{current_time}] Price: ${current_price:.4f} | Conf: {smc_data['confluence_score']:.2f} | Decision: {decision['decision']} | Quality: {decision['setup_quality']} | Swing bias: {smc_data['swing_bias']} | Volume: ${current_df['volume'].iloc[-1]:,.0f}"
                )
            if (
                decision["decision"] != "HOLD"
                and len(positions) < self.max_positions
                and (daily_trades_count < self.session_trades_limit)
                and (abs(daily_pnl) < balance * self.daily_loss_limit_pct)
            ):
                decision["confluence_score"] = smc_data["confluence_score"]
                decision["atr_value"] = decision.get(
                    "atr_value", compute_atr(current_df, 14).iloc[-1]
                )
                if not self._validate_trade_setup(
                    decision, current_price, smc_data, current_df
                ):
                    continue
                leverage = min(decision["leverage"], self.leverage)
                position_size = decision["position_size"]
                if position_size > 0:
                    (entry_price_with_slippage, commission) = (
                        self._apply_slippage_and_commission(
                            current_price,
                            position_size,
                            is_entry=True,
                            decision=decision["decision"],
                            df=current_df,
                        )
                    )
                    if (
                        entry_price_with_slippage <= 0
                        or decision["stop_loss"] <= 0
                        or decision["take_profit"] <= 0
                    ):
                        logger.warning(
                            f"Invalid trade parameters at {current_time}, skipping"
                        )
                        continue
                    stop_distance = abs(
                        entry_price_with_slippage - decision["stop_loss"]
                    )
                    reward_distance = (
                        abs(decision["take_profit"] - entry_price_with_slippage)
                        if decision["decision"] == "BUY"
                        else abs(entry_price_with_slippage - decision["take_profit"])
                    )
                    if (
                        stop_distance > 0
                        and reward_distance / stop_distance < self.strategy.min_rr_ratio
                    ):
                        logger.warning(
                            f"R:R ({reward_distance / stop_distance:.2f}) failed after slippage, skipping."
                        )
                        continue
                    position_id = f"{symbol}_{int(current_time.timestamp())}"
                    positions[position_id] = {
                        "asset_id": symbol,
                        "entry_time": current_time,
                        "entry_price": entry_price_with_slippage,
                        "position_size": position_size,
                        "decision": decision["decision"],
                        "stop_loss": decision["stop_loss"],
                        "take_profit": decision["take_profit"],
                        "leverage": leverage,
                        "commission": commission,
                        "reason": decision["entry_reason"],
                        "smc_context": {
                            "setup_quality": decision["setup_quality"],
                            "confluence_score": decision["confidence"],
                            "liquidity_sweep": decision.get("liquidity_sweep", False),
                            "risk_reward": decision["risk_reward"],
                            "zones": smc_data.get("zones", {}),
                            "swing_order_blocks": smc_data.get(
                                "swing_order_blocks", []
                            ),
                            "liquidity_pools": smc_data.get("liquidity_pools", []),
                            "trailing_extremes": smc_data.get("trailing_extremes", {}),
                        },
                        "risk_amount": balance * self.strategy.position_size_risk,
                        "max_adverse_excursion": 0.0,
                        "max_favorable_excursion": 0.0,
                    }
                    balance -= commission
                    total_commission += commission
                    total_trade_count += 1
                    daily_trades_count += 1
                    logger.info(
                        f"📈 Открыта позиция {decision['decision']} @ {entry_price_with_slippage:.4f} | Size: {position_size:.4f} | SL: {decision['stop_loss']:.4f} | TP: {decision['take_profit']:.4f} | Reason: {decision['entry_reason']} | Conf: {decision['confidence']:.2f}"
                    )
            closed_positions = []
            for pos_id, pos in positions.items():
                should_close = False
                exit_price = current_price
                exit_reason = ""
                candle_high = df["high"].iloc[i]
                candle_low = df["low"].iloc[i]
                entry_price_clean = pos["entry_price"]
                if pos["decision"] == "BUY":
                    mae = max(
                        pos.get("max_adverse_excursion", 0),
                        entry_price_clean - candle_low,
                    )
                    mfe = max(
                        pos.get("max_favorable_excursion", 0),
                        candle_high - entry_price_clean,
                    )
                else:
                    mae = max(
                        pos.get("max_adverse_excursion", 0),
                        candle_high - entry_price_clean,
                    )
                    mfe = max(
                        pos.get("max_favorable_excursion", 0),
                        entry_price_clean - candle_low,
                    )
                pos["max_adverse_excursion"] = mae
                pos["max_favorable_excursion"] = mfe
                if pos["decision"] == "BUY":
                    if candle_high >= pos["take_profit"]:
                        should_close = True
                        exit_price = min(candle_high, pos["take_profit"])
                        exit_reason = "TAKE_PROFIT"
                    elif candle_low <= pos["stop_loss"]:
                        should_close = True
                        current_open = df["open"].iloc[i]
                        exit_price = (
                            current_open
                            if current_open < pos["stop_loss"]
                            else pos["stop_loss"]
                        )
                        exit_reason = "STOP_LOSS"
                    elif (
                        smc_data["swing_bias"] == "BEARISH"
                        and smc_data["confluence_score"] <= -0.4
                        and (current_price < pos["entry_price"] * 0.99)
                    ):
                        should_close = True
                        exit_price = current_price
                        exit_reason = "TREND_REVERSAL"
                elif candle_low <= pos["take_profit"]:
                    should_close = True
                    exit_price = max(candle_low, pos["take_profit"])
                    exit_reason = "TAKE_PROFIT"
                elif candle_high >= pos["stop_loss"]:
                    should_close = True
                    exit_price = min(candle_high, pos["stop_loss"])
                    exit_reason = "STOP_LOSS"
                elif (
                    smc_data["swing_bias"] == "BULLISH"
                    and smc_data["confluence_score"] >= 0.4
                    and (current_price > pos["entry_price"] * 1.01)
                ):
                    should_close = True
                    exit_price = current_price
                    exit_reason = "TREND_REVERSAL"
                time_in_trade = (current_time - pos["entry_time"]).total_seconds()
                max_hold_time = 36 * 3600
                if not should_close and time_in_trade > max_hold_time:
                    is_profit = (
                        pos["decision"] == "BUY"
                        and current_price > pos["entry_price"]
                        or (
                            pos["decision"] == "SELL"
                            and current_price < pos["entry_price"]
                        )
                    )
                    if is_profit:
                        if time_in_trade > 48 * 3600:
                            should_close = True
                            exit_price = current_price
                            exit_reason = "TIME_EXPIRED_PROFIT"
                    else:
                        should_close = True
                        exit_price = current_price
                        exit_reason = "TIME_EXPIRED_STALE"
                if should_close:
                    (exit_price_with_slippage, commission) = (
                        self._apply_slippage_and_commission(
                            exit_price,
                            pos["position_size"],
                            is_entry=False,
                            decision=pos["decision"],
                            df=current_df,
                            position_type="exit",
                        )
                    )
                    pnl = self._calculate_pnl(
                        pos["entry_price"],
                        exit_price_with_slippage,
                        pos["position_size"],
                        pos["decision"],
                        pos["leverage"],
                        entry_commission=pos["commission"],
                        exit_commission=commission,
                        funding_payments=self._calculate_funding_payments(
                            pos["entry_time"],
                            current_time,
                            pos["position_size"],
                            pos["entry_price"],
                            pos["leverage"],
                            self.funding_rate,
                        ),
                    )
                    risk_amount = pos["risk_amount"]
                    position_value = (
                        pos["position_size"] * pos["entry_price"] * pos["leverage"]
                    )
                    pnl_pct_nominal = (
                        pnl / position_value * 100 if position_value > 0 else 0.0
                    )
                    actual_rr = (
                        pnl / risk_amount
                        if risk_amount > 0 and pnl > 0
                        else (pnl / abs(pnl) if pnl != 0 else 0) - 1
                    )
                    trade = TradeSimulation(
                        trade_id=pos_id,
                        entry_time=pos["entry_time"],
                        exit_time=current_time,
                        entry_price=pos["entry_price"],
                        exit_price=exit_price_with_slippage,
                        position_size=pos["position_size"],
                        decision=pos["decision"],
                        stop_loss=pos["stop_loss"],
                        take_profit=pos["take_profit"],
                        leverage=pos["leverage"],
                        pnl=pnl,
                        pnl_pct=pnl_pct_nominal,
                        holding_period=current_time - pos["entry_time"],
                        commission=commission + pos["commission"],
                        slippage=abs(exit_price - exit_price_with_slippage),
                        reason=exit_reason,
                        smc_context=pos["smc_context"],
                        entry_confidence=pos["smc_context"].get("confluence_score", 0),
                        exit_confidence=smc_data["confluence_score"],
                        risk_amount=risk_amount,
                        risk_reward_ratio=actual_rr,
                        max_adverse_excursion=pos.get("max_adverse_excursion", 0.0),
                        max_favorable_excursion=pos.get("max_favorable_excursion", 0.0),
                    )
                    trades.append(trade)
                    balance += pnl
                    total_commission += commission
                    daily_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
                    total_pnl += pnl
                    closed_positions.append(pos_id)
                    result_emoji = "✅" if pnl > 0 else "❌"
                    logger.info(
                        f"{result_emoji} Закрыта позиция {pos['decision']} | PnL: ${pnl:,.2f} | Причина: {exit_reason} | Удержание: {trade.holding_period}"
                    )
            for pos_id in closed_positions:
                del positions[pos_id]
            equity_curve.append(equity)
            timestamps.append(current_time)
            if equity > max_equity:
                max_equity = equity
            else:
                drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        for pos_id, pos in positions.items():
            exit_price = df["close"].iloc[-1]
            (exit_price_with_slippage, commission) = (
                self._apply_slippage_and_commission(
                    exit_price,
                    pos["position_size"],
                    is_entry=False,
                    decision=pos["decision"],
                    df=df.iloc[-1:].copy(),
                    position_type="exit",
                )
            )
            pnl = self._calculate_pnl(
                pos["entry_price"],
                exit_price_with_slippage,
                pos["position_size"],
                pos["decision"],
                pos["leverage"],
                entry_commission=pos["commission"],
                exit_commission=commission,
                funding_payments=self._calculate_funding_payments(
                    pos["entry_time"],
                    df.index[-1],
                    pos["position_size"],
                    pos["entry_price"],
                    pos["leverage"],
                    self.funding_rate,
                ),
            )
            risk_amount = pos["risk_amount"]
            position_value = pos["position_size"] * pos["entry_price"] * pos["leverage"]
            pnl_pct_nominal = pnl / position_value * 100 if position_value > 0 else 0.0
            actual_rr = (
                pnl / risk_amount
                if risk_amount > 0 and pnl > 0
                else (pnl / abs(pnl) if pnl != 0 else 0) - 1
            )
            trade = TradeSimulation(
                trade_id=pos_id,
                entry_time=pos["entry_time"],
                exit_time=df.index[-1],
                entry_price=pos["entry_price"],
                exit_price=exit_price_with_slippage,
                position_size=pos["position_size"],
                decision=pos["decision"],
                stop_loss=pos["stop_loss"],
                take_profit=pos["take_profit"],
                leverage=pos["leverage"],
                pnl=pnl,
                pnl_pct=pnl_pct_nominal,
                holding_period=df.index[-1] - pos["entry_time"],
                commission=commission + pos["commission"],
                slippage=abs(exit_price - exit_price_with_slippage),
                reason="END_OF_PERIOD",
                smc_context=pos["smc_context"],
                entry_confidence=pos["smc_context"].get("confluence_score", 0),
                exit_confidence=smc_data["confluence_score"] if smc_data else 0,
                risk_amount=risk_amount,
                risk_reward_ratio=actual_rr,
                max_adverse_excursion=pos.get("max_adverse_excursion", 0.0),
                max_favorable_excursion=pos.get("max_favorable_excursion", 0.0),
            )
            trades.append(trade)
            balance += pnl
            total_commission += commission
            daily_pnl += pnl
            if pnl > 0:
                winning_trades += 1
            total_pnl += pnl
            total_trade_count += 1
            result_emoji = "✅" if pnl > 0 else "❌"
            logger.info(
                f"{result_emoji} Закрыта позиция (конец периода) {pos['decision']} | PnL: ${pnl:,.2f}"
            )
        final_equity = balance + sum(
            (pos.get("unrealized_pnl", 0) for pos in positions.values())
        )
        total_return = (
            (final_equity - initial_balance) / initial_balance
            if initial_balance > 0
            else 0
        )
        duration_seconds = (end_time - start_time).total_seconds()
        duration_years = duration_seconds / (365.25 * 24 * 3600)
        annualized_return = (
            (1 + total_return) ** (1 / duration_years) - 1
            if duration_years > 0 and total_return > -1
            else 0
        )
        win_rate = winning_trades / total_trade_count if total_trade_count > 0 else 0
        winning_trades_list = [t for t in trades if t.pnl > 0]
        losing_trades_list = [t for t in trades if t.pnl < 0]
        gross_profit = (
            sum((t.pnl for t in winning_trades_list)) if winning_trades_list else 0
        )
        gross_loss = abs(sum((t.pnl for t in losing_trades_list)))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        if equity_curve and timestamps:
            equity_df = pd.Series(equity_curve, index=timestamps, name="equity")
            running_max = equity_df.cummax()
            drawdown_df = (running_max - equity_df) / running_max
            returns = equity_df.pct_change().dropna()
        else:
            equity_df = pd.Series([initial_balance], index=[start_time], name="equity")
            drawdown_df = pd.Series([0.0], index=[start_time], name="drawdown")
            returns = pd.Series()
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        recovery_factor = (
            total_return / max_drawdown if max_drawdown > 0 else float("inf")
        )
        trade_metrics = self._calculate_trade_metrics(trades)
        best_trade = max(trades, key=lambda x: x.pnl) if trades else None
        worst_trade = min(trades, key=lambda x: x.pnl) if trades else None
        result = BacktestResult(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            initial_balance=initial_balance,
            final_balance=float(final_equity),
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            total_trades=int(total_trade_count),
            winning_trades=int(winning_trades),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            max_drawdown=float(max_drawdown),
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            recovery_factor=float(recovery_factor),
            trades=trades,
            equity_curve=equity_df,
            drawdown_curve=drawdown_df,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=trade_metrics["avg_trade_duration"],
            avg_profit_per_trade=trade_metrics["avg_profit_per_trade"],
            avg_loss_per_trade=trade_metrics["avg_loss_per_trade"],
            largest_winning_trade=trade_metrics["largest_winning_trade"],
            largest_losing_trade=trade_metrics["largest_losing_trade"],
        )
        logger.info(
            f"✅ УЛУЧШЕННЫЙ бектест завершен | Финальный баланс: ${final_equity:,.2f} | Total Return: {total_return:.2%}"
        )
        logger.info(
            f"📊 Сделок: {total_trade_count} | Win Rate: {win_rate:.1%} | Profit Factor: {profit_factor:.2f}"
        )
        logger.info(
            f"📈 Sharpe Ratio: {sharpe_ratio:.2f} | Sortino Ratio: {sortino_ratio:.2f}"
        )
        logger.info(
            f"⚠️ Max Drawdown: {max_drawdown:.2%} | Recovery Factor: {recovery_factor:.2f}"
        )
        return result

    def plot_backtest_results(self, result: BacktestResult, file_path: Path):
        logger.info(f"🎨 Генерация Smart-графика: {file_path}")
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except:
            plt.style.use("bmh")
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 10,
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )
        df = self.bin_client.get_klines(
            symbol=result.symbol,
            interval=result.interval,
            start_time=int(result.start_time.timestamp() * 1000),
            end_time=int(result.end_time.timestamp() * 1000),
            limit=1500,
        )
        if df.empty:
            logger.warning("❌ Нет данных для отрисовки графика")
            return
        (fig, axes) = plt.subplots(
            3,
            1,
            figsize=(16, 12),
            sharex=True,
            gridspec_kw={"height_ratios": [5, 2, 1.5], "hspace": 0.05},
        )
        (ax_price, ax_equity, ax_dd) = axes
        self._plot_candles_clean(ax_price, df)
        self._plot_trades_clean(ax_price, result.trades)
        if result.trades:
            self._plot_key_levels_clean(ax_price, result.trades[-1].smc_context)
        title_text = f"{result.symbol} ({result.interval}) | Return: {result.total_return:+.2%} | Win Rate: {result.win_rate:.1%} | PF: {result.profit_factor:.2f}"
        ax_price.set_title(title_text, fontweight="bold", fontsize=12, pad=10)
        ax_price.set_ylabel("Price")
        if result.equity_curve is not None and (not result.equity_curve.empty):
            equity = result.equity_curve
            ax_equity.plot(
                equity.index,
                equity.values,
                color="#263238",
                linewidth=1.5,
                label="Equity",
            )
            ax_equity.axhline(
                result.initial_balance,
                color="gray",
                linestyle="--",
                alpha=0.8,
                linewidth=1,
            )
            ax_equity.fill_between(
                equity.index,
                result.initial_balance,
                equity.values,
                where=equity.values >= result.initial_balance,
                color="#66BB6A",
                alpha=0.3,
                interpolate=True,
            )
            ax_equity.fill_between(
                equity.index,
                result.initial_balance,
                equity.values,
                where=equity.values < result.initial_balance,
                color="#EF5350",
                alpha=0.3,
                interpolate=True,
            )
            ax_equity.set_ylabel("Equity ($)")
            ax_equity.legend(loc="upper left", fontsize=8)
        if result.drawdown_curve is not None and (not result.drawdown_curve.empty):
            dd = result.drawdown_curve * -100
            ax_dd.fill_between(dd.index, 0, dd.values, color="#C62828", alpha=0.4)
            ax_dd.plot(dd.index, dd.values, color="#B71C1C", linewidth=1)
            ax_dd.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            min_dd = dd.min()
            min_dd_date = dd.idxmin()
            if not np.isnan(min_dd):
                ax_dd.annotate(
                    f"Max DD: {min_dd:.1f}%",
                    xy=(min_dd_date, min_dd),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#B71C1C",
                    arrowprops=dict(facecolor="#B71C1C", arrowstyle="wedge", alpha=0.5),
                )
            ax_dd.set_ylabel("Drawdown %")
            ax_dd.set_ylim(bottom=min(dd.min() * 1.2, -5))
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        plt.xticks(rotation=0)
        ax_dd.set_xlabel("Date")
        file_path.parent.mkdir(exist_ok=True)
        plt.savefig(file_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"✅ График успешно сохранён: {file_path}")

    def _plot_candles_clean(self, ax, df):
        if len(df) > 1:
            time_diff = (df.index[1] - df.index[0]).total_seconds() / 86400
            width = time_diff * 0.7
        else:
            width = 0.04
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        col_up = "#089981"
        col_down = "#F23645"
        ax.bar(
            up.index,
            up.close - up.open,
            bottom=up.open,
            width=width,
            color=col_up,
            edgecolor=col_up,
            linewidth=0.5,
            zorder=3,
        )
        ax.bar(
            down.index,
            down.open - down.close,
            bottom=down.close,
            width=width,
            color=col_down,
            edgecolor=col_down,
            linewidth=0.5,
            zorder=3,
        )
        ax.vlines(up.index, up.low, up.high, color=col_up, linewidth=0.8, zorder=3)
        ax.vlines(
            down.index, down.low, down.high, color=col_down, linewidth=0.8, zorder=3
        )

    def _plot_trades_clean(self, ax, trades: List[TradeSimulation]):
        if not trades:
            return
        buys = [t for t in trades if t.decision == "BUY"]
        sells = [t for t in trades if t.decision == "SELL"]
        if buys:
            ax.scatter(
                [t.entry_time for t in buys],
                [t.entry_price for t in buys],
                marker="^",
                color="#2962FF",
                s=100,
                edgecolors="white",
                linewidth=1,
                zorder=10,
                label="Buy Entry",
            )
        if sells:
            ax.scatter(
                [t.entry_time for t in sells],
                [t.entry_price for t in sells],
                marker="v",
                color="#FF6D00",
                s=100,
                edgecolors="white",
                linewidth=1,
                zorder=10,
                label="Sell Entry",
            )
        for t in trades:
            color = "#00C853" if t.pnl > 0 else "#D50000"
            style = "--" if t.pnl > 0 else ":"
            ax.plot(
                [t.entry_time, t.exit_time],
                [t.entry_price, t.exit_price],
                color=color,
                linestyle=style,
                linewidth=1.2,
                alpha=0.8,
                zorder=5,
            )
            marker = "o" if t.pnl > 0 else "X"
            ax.scatter(
                [t.exit_time],
                [t.exit_price],
                marker=marker,
                color=color,
                s=50,
                edgecolors="white",
                linewidth=0.5,
                zorder=10,
            )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    def _plot_key_levels_clean(self, ax, smc_context):
        if not smc_context:
            return
        obs = smc_context.get("swing_order_blocks", [])
        valid_obs = [
            ob
            for ob in obs
            if isinstance(ob, dict) and "top" in ob and ("bottom" in ob)
        ]
        for ob in valid_obs[:5]:
            if not ob.get("mitigated", False):
                is_bullish = ob.get("type") == "Bullish" or ob.get("type") == 1
                color = "#2962FF" if is_bullish else "#FF6D00"
                ax.axhspan(ob["bottom"], ob["top"], alpha=0.1, color=color, zorder=1)
                border_lvl = ob["top"] if is_bullish else ob["bottom"]
                ax.axhline(
                    border_lvl,
                    color=color,
                    linestyle="-",
                    linewidth=0.5,
                    alpha=0.3,
                    zorder=1,
                )
        zones = smc_context.get("zones", {})
        if "discount" in zones and isinstance(zones["discount"], (list, tuple)):
            ax.axhline(
                zones["discount"][1],
                color="#00C853",
                linestyle="-.",
                linewidth=0.5,
                alpha=0.4,
                label="Discount Eq",
            )
        if "premium" in zones and isinstance(zones["premium"], (list, tuple)):
            ax.axhline(
                zones["premium"][0],
                color="#D50000",
                linestyle="-.",
                linewidth=0.5,
                alpha=0.4,
                label="Premium Eq",
            )


def generate_detailed_report(result: BacktestResult, report_path: Path):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SMART MONEY CONCEPTS ENHANCED BACKTEST REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"📊 Символ: {result.symbol}\n")
        f.write(f"⏱️ Интервал: {result.interval}\n")
        f.write(
            f"📅 Период: {result.start_time.strftime('%Y-%m-%d')} - {result.end_time.strftime('%Y-%m-%d')}\n"
        )
        f.write(f"💰 Начальный баланс: ${result.initial_balance:,.2f}\n")
        f.write(f"💰 Финальный баланс: ${result.final_balance:,.2f}\n")
        f.write(f"📈 Общая доходность: {result.total_return:.2%}\n")
        f.write(f"📅 Годовая доходность: {result.annualized_return:.2%}\n\n")
        f.write("=" * 80 + "\n")
        f.write("🎯 СТАТИСТИКА ПО СДЕЛКАМ\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"📊 Общее количество сделок: {result.total_trades}\n")
        f.write(f"✅ Количество прибыльных сделок: {result.winning_trades}\n")
        f.write(
            f"❌ Количество убыточных сделок: {result.total_trades - result.winning_trades}\n"
        )
        f.write(f"🎯 Win Rate: {result.win_rate:.1%}\n")
        f.write(f"💰 Profit Factor: {result.profit_factor:.2f}\n\n")
        f.write("=" * 80 + "\n")
        f.write("⚠️ РИСКОВЫЕ МЕТРИКИ\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"📉 Max Drawdown: {result.max_drawdown:.2%}\n")
        f.write(f"📈 Коэффициент восстановления: {result.recovery_factor:.2f}\n")
        f.write(f"📊 Коэффициент Шарпа: {result.sharpe_ratio:.2f}\n")
        f.write(f"📊 Коэффициент Сортино: {result.sortino_ratio:.2f}\n\n")
        if result.best_trade and result.worst_trade:
            f.write("=" * 80 + "\n")
            f.write("🏆 ЛУЧШАЯ И ХУДШАЯ СДЕЛКИ\n")
            f.write("=" * 80 + "\n\n")
            f.write("🥇 ЛУЧШАЯ СДЕЛКА:\n")
            f.write(f"   • Время входа: {result.best_trade.entry_time}\n")
            f.write(f"   • Время выхода: {result.best_trade.exit_time}\n")
            f.write(f"   • Направление: {result.best_trade.decision}\n")
            f.write(f"   • Цена входа: ${result.best_trade.entry_price:.4f}\n")
            f.write(f"   • Цена выхода: ${result.best_trade.exit_price:.4f}\n")
            f.write(
                f"   • PnL: ${result.best_trade.pnl:,.2f} ({result.best_trade.pnl_pct:.2f}%)\n"
            )
            f.write(f"   • Причина закрытия: {result.best_trade.reason}\n")
            f.write(f"   • Период удержания: {result.best_trade.holding_period}\n\n")
            f.write("🥈 ХУДШАЯ СДЕЛКА:\n")
            f.write(f"   • Время входа: {result.worst_trade.entry_time}\n")
            f.write(f"   • Время выхода: {result.worst_trade.exit_time}\n")
            f.write(f"   • Направление: {result.worst_trade.decision}\n")
            f.write(f"   • Цена входа: ${result.worst_trade.entry_price:.4f}\n")
            f.write(f"   • Цена выхода: ${result.worst_trade.exit_price:.4f}\n")
            f.write(
                f"   • PnL: ${result.worst_trade.pnl:,.2f} ({result.worst_trade.pnl_pct:.2f}%)\n"
            )
            f.write(f"   • Причина закрытия: {result.worst_trade.reason}\n")
            f.write(f"   • Период удержания: {result.worst_trade.holding_period}\n\n")
        f.write("=" * 80 + "\n")
        f.write("🔍 АНАЛИЗ КАЧЕСТВА СИГНАЛОВ\n")
        f.write("=" * 80 + "\n\n")
        if result.trades:
            quality_stats = {}
            for trade in result.trades:
                quality = trade.smc_context.get("setup_quality", "UNKNOWN")
                if quality not in quality_stats:
                    quality_stats[quality] = {"count": 0, "pnl": 0, "win_count": 0}
                quality_stats[quality]["count"] += 1
                quality_stats[quality]["pnl"] += trade.pnl
                if trade.pnl > 0:
                    quality_stats[quality]["win_count"] += 1
            for quality, stats in quality_stats.items():
                win_rate = (
                    stats["win_count"] / stats["count"] if stats["count"] > 0 else 0
                )
                avg_pnl = stats["pnl"] / stats["count"] if stats["count"] > 0 else 0
                f.write(f"{quality.upper()} сигналы:\n")
                f.write(f"   • Количество: {stats['count']}\n")
                f.write(f"   • Win Rate: {win_rate:.1%}\n")
                f.write(f"   • Средний PnL: ${avg_pnl:,.2f}\n")
                f.write(f"   • Общий PnL: ${stats['pnl']:,.2f}\n\n")
        f.write("=" * 80 + "\n")
        f.write("💡 ЗАКЛЮЧЕНИЕ\n")
        f.write("=" * 80 + "\n\n")
        if result.total_return > 0.1:
            f.write("✅ СИСТЕМА ПОКАЗАЛА ОТЛИЧНЫЕ РЕЗУЛЬТАТЫ!\n")
            f.write(
                "Рекомендуется протестировать на реальном счете с небольшими объемами.\n"
            )
        elif result.total_return > 0:
            f.write("🟡 СИСТЕМА ПОКАЗАЛА УМЕРЕННЫЕ РЕЗУЛЬТАТЫ\n")
            f.write(
                "Рекомендуется оптимизация параметров и дополнительное тестирование.\n"
            )
        else:
            f.write("❌ СИСТЕМА НЕ ПОКАЗАЛА ПРИБЫЛЬНОСТИ\n")
            f.write(
                "Требуется глубокая доработка логики сигналов и риск-менеджмента.\n"
            )
        f.write(
            f"\nОтчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )


def generate_trades_summary(result: BacktestResult, summary_path: Path):
    if not result.trades:
        return
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trade_id",
                "entry_time",
                "exit_time",
                "decision",
                "entry_price",
                "exit_price",
                "position_size",
                "leverage",
                "stop_loss",
                "take_profit",
                "pnl",
                "pnl_pct",
                "holding_period",
                "reason",
                "setup_quality",
                "confluence_score",
                "liquidity_sweep",
            ]
        )
        for trade in result.trades:
            smc_context = trade.smc_context
            writer.writerow(
                [
                    trade.trade_id,
                    trade.entry_time,
                    trade.exit_time,
                    trade.decision,
                    trade.entry_price,
                    trade.exit_price,
                    trade.position_size,
                    trade.leverage,
                    trade.stop_loss,
                    trade.take_profit,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.holding_period,
                    trade.reason,
                    smc_context.get("setup_quality", "N/A"),
                    smc_context.get("confluence_score", 0),
                    smc_context.get("liquidity_sweep", False),
                ]
            )


def generate_recommendations(result: BacktestResult) -> List[str]:
    recommendations = []
    if result.total_return < 0:
        recommendations.append(
            "Система показала убыточные результаты. Требуется пересмотр логики генерации сигналов."
        )
    elif result.total_return < 0.05:
        recommendations.append(
            "Доходность ниже 5% за 3 месяца. Рассмотрите оптимизацию параметров стратегии."
        )
    elif result.total_return > 0.2:
        recommendations.append(
            "Отличная доходность! Рассмотрите возможность тестирования на других активах."
        )
    if result.win_rate < 0.4:
        recommendations.append(
            "Низкий Win Rate (<40%). Усильте фильтрацию сигналов и добавьте дополнительные подтверждения."
        )
    elif result.win_rate > 0.6:
        recommendations.append(
            "Высокий Win Rate (>60%). Система показывает стабильные результаты."
        )
    if result.profit_factor < 1.2:
        recommendations.append(
            "Profit Factor ниже 1.2. Увеличьте соотношение прибыльных/убыточных сделок."
        )
    elif result.profit_factor > 2.0:
        recommendations.append(
            "Отличный Profit Factor (>2.0). Система эффективно управляет рисками."
        )
    if result.max_drawdown > 0.2:
        recommendations.append(
            "Высокий Max Drawdown (>20%). Усильте управление позициями и добавьте динамическое регулирование риска."
        )
    elif result.max_drawdown < 0.1:
        recommendations.append(
            "Низкий Max Drawdown (<10%). Система показывает отличное управление рисками."
        )
    high_quality_trades = [
        t for t in result.trades if t.smc_context.get("setup_quality") == "HIGH"
    ]
    if high_quality_trades and len(high_quality_trades) > 5:
        high_quality_win_rate = len(
            [t for t in high_quality_trades if t.pnl > 0]
        ) / len(high_quality_trades)
        if high_quality_win_rate < 0.5:
            recommendations.append(
                "HIGH качество сигналов не подтверждается результатами. Пересмотрите критерии качества сигналов."
            )
    if result.sharpe_ratio < 0.5:
        recommendations.append(
            "Низкий коэффициент Шарпа. Добавьте фильтрацию по волатильности и временные фильтры."
        )
    if not recommendations:
        recommendations.append(
            "Система показывает сбалансированные результаты. Продолжайте мониторинг и постепенную оптимизацию."
        )
    return recommendations


def run_enhanced_backtest():
    print("=" * 60)
    print("🚀 SMART MONEY CONCEPTS ENHANCED BACKTESTING SYSTEM v3.0")
    print("=" * 60)
    config = {
        "use_futures": True,
        "internal_confluence_filter": False,
        "volatility_method": "ATR",
        "atr_length": 50,
        "ob_mitigation": "HIGHLOW",
        "min_volume_threshold": 50000.0,
        "max_correlation_risk": 0.5,
        "position_sizing_method": "risk_based",
        "risk_per_trade": 0.025,
        "commission_rate": 0.0004,
        "slippage_pct": 0.0005,
        "max_positions": 5,
        "leverage": 3,
        "time_filters": {
            "avoid_low_volume": [],
            "avoid_market_open": [],
            "prefer_high_liquidity": [],
        },
        "session_trades_limit": 30,
        "daily_loss_limit_pct": 0.15,
        "allow_low_liquidity": True,
        "enable_debug_logging": True,
    }
    symbol = "SOLUSDT"
    interval = "1h"
    end_time = datetime.now(timezone.utc) - timedelta(days=1)
    start_time = end_time - timedelta(days=90)
    print(f"📊 Символ: {symbol}")
    print(f"⏱️ Интервал: {interval}")
    print(
        f"📅 Период: {start_time.strftime('%Y-%m-%d')} - {end_time.strftime('%Y-%m-%d')}"
    )
    print("-" * 60)
    engine = EnhancedBacktestEngine(config)
    try:
        start_time_test = time.time()
        result = engine.run_enhanced_backtest(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            initial_balance=10000.0,
        )
        end_time_test = time.time()
        test_duration = end_time_test - start_time_test
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = BACKTEST_DIR / f"enhanced_{symbol}_{interval}_{timestamp}"
        result_dir.mkdir(exist_ok=True)
        result_path = result_dir / "results.json"
        result.to_json(result_path)
        report_path = result_dir / "detailed_report.txt"
        generate_detailed_report(result, report_path)
        chart_path = result_dir / "enhanced_backtest_chart.png"
        engine.plot_backtest_results(result, chart_path)
        trades_summary_path = result_dir / "trades_summary.csv"
        generate_trades_summary(result, trades_summary_path)
        print("-" * 60)
        print(f"✅ УЛУЧШЕННЫЙ бектест успешно завершен за {test_duration:.2f} секунд!")
        print(f"📁 Результаты сохранены в: {result_dir}")
        print(f"💰 Финальный баланс: ${result.final_balance:,.2f}")
        print(f"📈 Общая доходность: {result.total_return:.2%}")
        print(
            f"🎯 Win Rate: {result.win_rate:.1%} ({result.winning_trades}/{result.total_trades} сделок)"
        )
        print(f"📊 Profit Factor: {result.profit_factor:.2f}")
        print(f"⚠️ Max Drawdown: {result.max_drawdown:.2%}")
        print(
            f"📈 Sharpe Ratio: {result.sharpe_ratio:.2f} | Sortino Ratio: {result.sortino_ratio:.2f}"
        )
        if result.trades:
            avg_win = (
                np.mean([t.pnl for t in result.trades if t.pnl > 0])
                if any((t.pnl > 0 for t in result.trades))
                else 0
            )
            avg_loss = (
                np.mean([abs(t.pnl) for t in result.trades if t.pnl < 0])
                if any((t.pnl < 0 for t in result.trades))
                else 0
            )
            print(
                f"💰 Средний выигрыш: ${avg_win:,.2f} | Средний убыток: ${avg_loss:,.2f}"
            )
            print(
                f"⚖️ Profit/Loss Ratio: {(avg_win / avg_loss if avg_loss > 0 else 0):.2f}"
            )
        print("\n🎯 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ:")
        recommendations = generate_recommendations(result)
        for rec in recommendations:
            print(f"• {rec}")
    except Exception as e:
        logger.error(
            f"❌ Критическая ошибка при выполнении улучшенного бектеста: {e}",
            exc_info=True,
        )
        print(f"❌ Ошибка: {e}")


def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    json_config = load_config("config_smc.json")

    flat_config = {
        **json_config,
        **json_config["strategy_settings"],
        **json_config["risk_management"],
        **json_config["execution_settings"],
        "time_filters": json_config["time_filters"],
    }

    engine = EnhancedBacktestEngine(flat_config)
    run_enhanced_backtest()
