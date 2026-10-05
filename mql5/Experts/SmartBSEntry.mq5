//+------------------------------------------------------------------+
//| SmartBSEntry.mq5 — chart parity: all_blend + exit_arm + ST filter|
//| pivot_breakout L=15 V2 | XAU/XAG/XTI/BTC/ETH                     |
//+------------------------------------------------------------------+
#property copyright "SmartBS"
#property version   "3.50"
#property strict
#property description "Chart parity: all_blend + exit_arm(SMA14) + ST filter(ATR15x3). pivot_breakout L=15 V2."

// ONNX + JSON sidecars for Strategy Tester (MQL5/Files/SmartBSEntry/).
// Symbols: XAUUSD XAGUSD XTIUSD BTCUSD ETHUSD × all trained engines.
#property tester_file "SmartBSEntry\\XAUUSD_all_blend.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_all_blend.json"
#property tester_file "SmartBSEntry\\XAUUSD_candle.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_candle.json"
#property tester_file "SmartBSEntry\\XAUUSD_common.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_common.json"
#property tester_file "SmartBSEntry\\XAUUSD_dbb.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_dbb.json"
#property tester_file "SmartBSEntry\\XAUUSD_macd.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_macd.json"
#property tester_file "SmartBSEntry\\XAUUSD_maribbon.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_maribbon.json"
#property tester_file "SmartBSEntry\\XAUUSD_pivot_engine.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_pivot_engine.json"
#property tester_file "SmartBSEntry\\XAUUSD_regime_engine.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_regime_engine.json"
#property tester_file "SmartBSEntry\\XAUUSD_rsi_divergence.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_rsi_divergence.json"
#property tester_file "SmartBSEntry\\XAUUSD_signals.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_signals.json"
#property tester_file "SmartBSEntry\\XAUUSD_smart_money.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_smart_money.json"
#property tester_file "SmartBSEntry\\XAUUSD_trend_pullback.onnx"
#property tester_file "SmartBSEntry\\XAUUSD_trend_pullback.json"
#property tester_file "SmartBSEntry\\XAGUSD_all_blend.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_all_blend.json"
#property tester_file "SmartBSEntry\\XAGUSD_candle.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_candle.json"
#property tester_file "SmartBSEntry\\XAGUSD_common.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_common.json"
#property tester_file "SmartBSEntry\\XAGUSD_dbb.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_dbb.json"
#property tester_file "SmartBSEntry\\XAGUSD_macd.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_macd.json"
#property tester_file "SmartBSEntry\\XAGUSD_maribbon.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_maribbon.json"
#property tester_file "SmartBSEntry\\XAGUSD_pivot_engine.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_pivot_engine.json"
#property tester_file "SmartBSEntry\\XAGUSD_regime_engine.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_regime_engine.json"
#property tester_file "SmartBSEntry\\XAGUSD_rsi_divergence.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_rsi_divergence.json"
#property tester_file "SmartBSEntry\\XAGUSD_signals.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_signals.json"
#property tester_file "SmartBSEntry\\XAGUSD_smart_money.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_smart_money.json"
#property tester_file "SmartBSEntry\\XAGUSD_trend_pullback.onnx"
#property tester_file "SmartBSEntry\\XAGUSD_trend_pullback.json"
#property tester_file "SmartBSEntry\\XTIUSD_all_blend.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_all_blend.json"
#property tester_file "SmartBSEntry\\XTIUSD_candle.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_candle.json"
#property tester_file "SmartBSEntry\\XTIUSD_common.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_common.json"
#property tester_file "SmartBSEntry\\XTIUSD_dbb.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_dbb.json"
#property tester_file "SmartBSEntry\\XTIUSD_macd.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_macd.json"
#property tester_file "SmartBSEntry\\XTIUSD_maribbon.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_maribbon.json"
#property tester_file "SmartBSEntry\\XTIUSD_pivot_engine.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_pivot_engine.json"
#property tester_file "SmartBSEntry\\XTIUSD_regime_engine.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_regime_engine.json"
#property tester_file "SmartBSEntry\\XTIUSD_rsi_divergence.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_rsi_divergence.json"
#property tester_file "SmartBSEntry\\XTIUSD_signals.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_signals.json"
#property tester_file "SmartBSEntry\\XTIUSD_smart_money.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_smart_money.json"
#property tester_file "SmartBSEntry\\XTIUSD_trend_pullback.onnx"
#property tester_file "SmartBSEntry\\XTIUSD_trend_pullback.json"
#property tester_file "SmartBSEntry\\BTCUSD_all_blend.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_all_blend.json"
#property tester_file "SmartBSEntry\\BTCUSD_candle.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_candle.json"
#property tester_file "SmartBSEntry\\BTCUSD_common.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_common.json"
#property tester_file "SmartBSEntry\\BTCUSD_dbb.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_dbb.json"
#property tester_file "SmartBSEntry\\BTCUSD_macd.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_macd.json"
#property tester_file "SmartBSEntry\\BTCUSD_maribbon.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_maribbon.json"
#property tester_file "SmartBSEntry\\BTCUSD_pivot_engine.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_pivot_engine.json"
#property tester_file "SmartBSEntry\\BTCUSD_regime_engine.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_regime_engine.json"
#property tester_file "SmartBSEntry\\BTCUSD_rsi_divergence.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_rsi_divergence.json"
#property tester_file "SmartBSEntry\\BTCUSD_signals.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_signals.json"
#property tester_file "SmartBSEntry\\BTCUSD_smart_money.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_smart_money.json"
#property tester_file "SmartBSEntry\\BTCUSD_trend_pullback.onnx"
#property tester_file "SmartBSEntry\\BTCUSD_trend_pullback.json"
#property tester_file "SmartBSEntry\\ETHUSD_all_blend.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_all_blend.json"
#property tester_file "SmartBSEntry\\ETHUSD_candle.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_candle.json"
#property tester_file "SmartBSEntry\\ETHUSD_common.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_common.json"
#property tester_file "SmartBSEntry\\ETHUSD_dbb.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_dbb.json"
#property tester_file "SmartBSEntry\\ETHUSD_macd.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_macd.json"
#property tester_file "SmartBSEntry\\ETHUSD_maribbon.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_maribbon.json"
#property tester_file "SmartBSEntry\\ETHUSD_pivot_engine.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_pivot_engine.json"
#property tester_file "SmartBSEntry\\ETHUSD_regime_engine.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_regime_engine.json"
#property tester_file "SmartBSEntry\\ETHUSD_rsi_divergence.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_rsi_divergence.json"
#property tester_file "SmartBSEntry\\ETHUSD_signals.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_signals.json"
#property tester_file "SmartBSEntry\\ETHUSD_smart_money.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_smart_money.json"
#property tester_file "SmartBSEntry\\ETHUSD_trend_pullback.onnx"
#property tester_file "SmartBSEntry\\ETHUSD_trend_pullback.json"
#include <Trade\Trade.mqh>
#include <SmartBSEntry\OnnxModel.mqh>
#include <SmartBSEntry\FeatureDispatch.mqh>
#include <SmartBSEntry\Softmax.mqh>
#include <SmartBSEntry\RiskManager.mqh>

#define SB_MAX_BLEND 12

enum ENUM_SB_SIGNAL_MODE
  {
   SB_MODE_RAW_AI       = 0, // argmax only
   SB_MODE_AI_THRESHOLD = 1  // argmax + min conf -> else FLAT
  };

enum ENUM_SB_TRADE_POLICY
  {
   SB_TRADE_SESSION = 0, // session_trade: decide @ session open, hold to end
   SB_TRADE_BAR     = 1  // every H1: AI-only / onset / MA gates (default)
  };

enum ENUM_SB_SESSION_HOURS
  {
   SB_INP_SESS_NORMAL   = 0, // Tokyo 0-9 / London 7-16 / NY 12-21 (default)
   SB_INP_SESS_ADJUSTED = 1  // Tokyo 0-7 / London 7-12 / NY 12-20
  };

// Engines: all_blend,candle,common,dbb,macd,maribbon,pivot_engine,regime_engine,
//          rsi_divergence,signals,smart_money,trend_pullback  (comma = avg blend)
input string              InpEngines      = "all_blend"; // pivot_breakout L=15 V2 (115ch)
input string              InpOnnxDir      = "SmartBSEntry\\"; // under MQL5/Files/
input double              InpTemperature  = 0.0;       // 0 = sidecar JSON (match Python)
input ENUM_SB_SIGNAL_MODE InpSignalMode   = SB_MODE_RAW_AI; // KEEP raw_ai
input double              InpAiThreshold  = 0.0;       // ignored under raw_ai
input int                 InpLookback     = 64;
input bool                InpOnlyClosedBar = true;
input bool                InpTradeRawAi   = true;      // must stay true for tester/live sync
input ENUM_SB_TRADE_POLICY InpTradePolicy = SB_TRADE_BAR; // every-bar = Python/chart replay
input ENUM_SB_SESSION_HOURS InpSessionHours = SB_INP_SESS_NORMAL; // match Python --session-hours
input double              InpLot          = 0.0;       // 0 = auto per asset (XAU0.1/XAG0.5/XTI0.1/BTC0.01/ETH0.1)
input ulong               InpMagic        = 260911;
input int                 InpSlippage     = 30;
// --- Chart parity (index.html: exit_arm ON + ST filter ON) ---
input bool                InpExitArm      = true;      // ON — hold until SMA14 confirms exit on AI FLAT/flip
input bool                InpStFilter     = true;      // ON — long only ST bull; short only ST bear (clip)
input int                 InpStAtrLen     = 15;        // match chart pivot_len ATR
input double              InpStFactor     = 3.0;       // match chart ST factor
// --- Bar policy gates (OFF under chart parity) ---
input bool                InpSignalPointGate = false;  // OFF
input double              InpSignalPointThr  = 0.35;
input bool                InpSignalPointAgree = true;
input bool                InpMaAiGate     = false;     // OFF (exit_arm is separate)
input int                 InpFastMaLen    = 14;        // SMA for exit_arm / MA gate
// --- Swing SL / BE / ladder (OFF = chart exit_arm+ST path) ---
input bool                InpUseStopLoss  = false;     // OFF
input bool                InpSLBE         = false;
input bool                InpSLLadder     = false;
input int                 InpSwingLen     = 9;         // pivot L/R
input int                 InpLadderMaxR   = 20;
input double              InpBeOffsetR    = 0.08;
input bool                InpValidateR    = false;     // OFF = geometry-only swing SL (no chase/ATR/pct)
input double              InpRMinAtr      = 0.25;      // used only if InpValidateR
input double              InpRMaxAtr      = 3.0;
input double              InpRMaxPct      = 0.012;
input double              InpChaseFrac    = 0.45;
input bool                InpRequireStop  = false;     // skip/open-close if SL cannot place

CSBOnnxModel g_models[SB_MAX_BLEND];
string       g_engines[SB_MAX_BLEND];
int          g_n_engines = 0;
int          g_ready_n = 0;
datetime     g_last_bar = 0;
string       g_asset = "XAUUSD";
double       g_trade_lot = 0.01;
double       g_last_probs[3];
int          g_last_cls = 0;
string       g_last_reason = "";
CTrade       g_trade;
ulong        g_pos_ticket = 0;
double       g_entry = 0.0;
double       g_initial_sl = 0.0;
double       g_r = 0.0;
int          g_rung = 0;
// Session-trade state (locked side for current session window)
int          g_sess_id = SB_SESS_NONE;
datetime     g_sess_start_gmt = 0;
int          g_sess_locked_cls = 0; // 0/1/2 at decision bar
bool         g_sess_has_lock = false;
// exit_arm logical book (±1/0). Broker pos may be flat under ST filter while this stays open.
int          g_logical_have = 0;

string SB_AssetStemFromChart(void)
  {
   string s = _Symbol;
   StringToUpper(s);
   // strip broker suffixes: BTCUSD.a / EURUSD.m
   int dot = StringFind(s, ".");
   if(dot > 0)
      s = StringSubstr(s, 0, dot);

   // commodities
   if(StringFind(s, "XAU") >= 0 || StringFind(s, "GOLD") >= 0)
      return "XAUUSD";
   if(StringFind(s, "XAG") >= 0 || StringFind(s, "SILVER") >= 0)
      return "XAGUSD";
   if(StringFind(s, "XTI") >= 0 || StringFind(s, "USOIL") >= 0 || StringFind(s, "WTI") >= 0)
      return "XTIUSD";
   if(StringFind(s, "NATGAS") >= 0 || StringFind(s, "XNG") >= 0 || StringFind(s, "NGAS") >= 0)
      return "NATGAS";
   if(StringFind(s, "XPT") >= 0 || StringFind(s, "PLAT") >= 0)
      return "PLATINUM";

   // crypto
   if(StringFind(s, "BTC") >= 0)
      return "BTCUSD";
   if(StringFind(s, "ETH") >= 0)
      return "ETHUSD";
   if(StringFind(s, "LTC") >= 0)
      return "LTCUSD";
   if(StringFind(s, "ADA") >= 0)
      return "ADAUSD";
   if(StringFind(s, "SOL") >= 0)
      return "SOLUSD";

   // forex
   if(StringFind(s, "EUR") >= 0 && StringFind(s, "USD") >= 0)
      return "EURUSD";
   if(StringFind(s, "GBP") >= 0 && StringFind(s, "USD") >= 0)
      return "GBPUSD";
   if(StringFind(s, "AUD") >= 0 && StringFind(s, "USD") >= 0)
      return "AUDUSD";
   if(StringFind(s, "USD") >= 0 && StringFind(s, "CHF") >= 0)
      return "USDCHF";
   if(StringFind(s, "USD") >= 0 && StringFind(s, "JPY") >= 0)
      return "USDJPY";

   return s;
  }

double SB_AutoLotForAsset(const string asset)
  {
   // Match Python replay LOTS in _train_replay_all_st / chart meta.
   if(asset == "XAUUSD" || asset == "XTIUSD" || asset == "ETHUSD")
      return 0.1;
   if(asset == "XAGUSD")
      return 0.5;
   if(asset == "BTCUSD")
      return 0.01;
   if(asset == "EURUSD" || asset == "GBPUSD" || asset == "AUDUSD" ||
      asset == "USDCHF" || asset == "USDJPY")
      return 0.1;
   return 0.01;
  }

double SB_NormalizeLot(double lot)
  {
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0)
      step = 0.01;
   if(vmin <= 0.0)
      vmin = step;
   lot = MathFloor(lot / step + 1e-12) * step;
   if(lot < vmin)
      lot = vmin;
   if(vmax > 0.0 && lot > vmax)
      lot = vmax;
   return lot;
  }

double SB_TradeLot(void)
  {
   double lot = (InpLot > 0.0) ? InpLot : SB_AutoLotForAsset(g_asset);
   return SB_NormalizeLot(lot);
  }

string SB_DefaultOnnxPath(const string asset, const string engine)
  {
   return InpOnnxDir + asset + "_" + engine + ".onnx";
  }

bool SB_LoadSidecar(const string onnx_path, double &temp, int &num_inputs, int &lookback)
  {
   string json_path = onnx_path;
   int dot = StringLen(json_path) - 5;
   if(dot > 0 && StringSubstr(json_path, dot) == ".onnx")
      json_path = StringSubstr(json_path, 0, dot) + ".json";
   else
      json_path = onnx_path + ".json";

   int h = FileOpen(json_path, FILE_READ | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
      h = FileOpen(json_path, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
      return false;

   string body = "";
   while(!FileIsEnding(h))
      body += FileReadString(h);
   FileClose(h);

   int p;
   p = StringFind(body, "\"temperature\"");
   if(p >= 0)
     {
      p = StringFind(body, ":", p);
      if(p >= 0)
         temp = StringToDouble(StringSubstr(body, p + 1));
     }
   p = StringFind(body, "\"num_inputs\"");
   if(p >= 0)
     {
      p = StringFind(body, ":", p);
      if(p >= 0)
         num_inputs = (int)StringToInteger(StringSubstr(body, p + 1));
     }
   p = StringFind(body, "\"lookback\"");
   if(p >= 0)
     {
      p = StringFind(body, ":", p);
      if(p >= 0)
         lookback = (int)StringToInteger(StringSubstr(body, p + 1));
     }
   return true;
  }

string SB_ClassName(const int cls)
  {
   if(cls == 1)
      return "LONG";
   if(cls == 2)
      return "SHORT";
   return "FLAT";
  }

void SB_UpdateComment(void)
  {
   string eng_list = "";
   for(int i = 0; i < g_n_engines; i++)
     {
      if(i > 0)
         eng_list += ",";
      eng_list += g_engines[i];
      if(!g_models[i].Ready())
         eng_list += "!";
     }
   string policy = "AI-only";
   if(InpTradePolicy == SB_TRADE_SESSION)
      policy = "session_trade";
   else if(InpExitArm || InpStFilter)
      policy = (InpExitArm && InpStFilter) ? "exit_arm+ST" :
               (InpExitArm ? "exit_arm" : "ST");
   else if(InpMaAiGate)
      policy = "MA+AI";
   else if(InpSignalPointGate)
      policy = "signal_point";
   string sess = "";
   if(InpTradePolicy == SB_TRADE_SESSION)
      sess = StringFormat(" | %s lock=%s", SB_SessionName(g_sess_id), SB_ClassName(g_sess_locked_cls));
   else if(InpExitArm || InpStFilter)
      sess = StringFormat(" | logical=%s", (g_logical_have > 0 ? "LONG" : (g_logical_have < 0 ? "SHORT" : "FLAT")));
   string line = StringFormat(
      "SmartBS %d/%d | %s | lot=%.2f | %s | F=%.3f L=%.3f S=%.3f | %s%s | trade=%s%s",
      g_ready_n, g_n_engines, g_asset, g_trade_lot, SB_ClassName(g_last_cls),
      g_last_probs[0], g_last_probs[1], g_last_probs[2],
      policy,
      (StringLen(g_last_reason) > 0 ? " @" + g_last_reason : ""),
      (InpTradeRawAi ? "ON" : "OFF"),
      sess);
   Comment(line + "\n" + eng_list);
  }

int SB_CurrentPosDir(void)
  {
   int dir = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!PositionSelectByTicket(PositionGetTicket(i)))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      long type = PositionGetInteger(POSITION_TYPE);
      if(type == POSITION_TYPE_BUY)
         dir = 1;
      else if(type == POSITION_TYPE_SELL)
         dir = -1;
     }
   return dir;
  }

ulong SB_OurTicket(void)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      return ticket;
     }
   return 0;
  }

double SB_NormPrice(const double price)
  {
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   return NormalizeDouble(price, digits);
  }

void SB_ClearLadderState(void)
  {
   g_pos_ticket = 0;
   g_entry = 0.0;
   g_initial_sl = 0.0;
   g_r = 0.0;
   g_rung = 0;
  }

void SB_InitLadderState(const ulong ticket, const int side, const double sl)
  {
   if(!PositionSelectByTicket(ticket))
      return;
   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   double r = MathAbs(entry - sl);
   if(r <= 0.0 || !MathIsValidNumber(r))
      return;
   g_pos_ticket = ticket;
   g_entry = entry;
   g_initial_sl = sl;
   g_r = r;
   g_rung = 0;
   PrintFormat("Ladder init ticket=%I64u entry=%.5f initSL=%.5f 1R=%.5f side=%s",
               ticket, entry, sl, r, (side > 0 ? "LONG" : "SHORT"));
  }

bool SB_SlOnProtectiveSide(const int side, const double sl)
  {
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(side > 0)
      return (sl < bid);
   return (sl > ask);
  }

bool SB_ModifySL(const ulong ticket, const double new_sl, const string tag)
  {
   if(!PositionSelectByTicket(ticket))
      return false;
   double cur_sl = PositionGetDouble(POSITION_SL);
   double tp = PositionGetDouble(POSITION_TP);
   long type = PositionGetInteger(POSITION_TYPE);
   int side = (type == POSITION_TYPE_BUY) ? 1 : -1;
   double sl = SB_NormPrice(new_sl);

   if(!SB_SlOnProtectiveSide(side, sl))
      return false;

   if(cur_sl > 0.0)
     {
      if(side > 0 && sl <= cur_sl + (_Point * 0.5))
         return true;
      if(side < 0 && sl >= cur_sl - (_Point * 0.5))
         return true;
     }

   if(!g_trade.PositionModify(ticket, sl, tp))
     {
      PrintFormat("%s modify failed ticket=%I64u sl=%.5f err=%d",
                  tag, ticket, sl, GetLastError());
      return false;
     }
   PrintFormat("%s %s SL %.5f -> %.5f",
               tag, (side > 0 ? "LONG" : "SHORT"), cur_sl, sl);
   return true;
  }

bool SB_AttachSwingSL(const int side, const ulong ticket)
  {
   if(!InpUseStopLoss)
      return true;
   if(!PositionSelectByTicket(ticket))
      return false;
   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   string serr;
   double sl = 0.0, r = 0.0;
   if(!SB_MakeSwingStop(side, entry, InpSwingLen, InpRMinAtr, InpRMaxAtr,
                        InpRMaxPct, InpChaseFrac, sl, r, serr, InpValidateR))
     {
      Print("Swing SL attach skip: ", serr);
      return false;
     }
   if(!SB_SlOnProtectiveSide(side, sl))
     {
      PrintFormat("Swing SL %.5f wrong side of market — defer", sl);
      return false;
     }
   if(!SB_ModifySL(ticket, sl, "InitSwingSL"))
      return false;
   SB_InitLadderState(ticket, side, sl);
   return true;
  }

void SB_EnsureSwingSL(void)
  {
   if(!InpUseStopLoss)
      return;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      if(PositionGetDouble(POSITION_SL) > 0.0)
        {
         if(g_r <= 0.0 || g_pos_ticket != ticket)
           {
            long type = PositionGetInteger(POSITION_TYPE);
            int side = (type == POSITION_TYPE_BUY) ? 1 : -1;
            SB_InitLadderState(ticket, side, PositionGetDouble(POSITION_SL));
           }
         continue;
        }
      long type = PositionGetInteger(POSITION_TYPE);
      int side = (type == POSITION_TYPE_BUY) ? 1 : -1;
      SB_AttachSwingSL(side, ticket);
     }
  }

void SB_UpdateSLBELadder(void)
  {
   if(!InpUseStopLoss)
      return;
   if(!InpSLBE && !InpSLLadder)
      return;
   if(g_r <= 0.0 || g_pos_ticket == 0)
      return;
   if(!PositionSelectByTicket(g_pos_ticket))
     {
      SB_ClearLadderState();
      return;
     }

   long type = PositionGetInteger(POSITION_TYPE);
   int side = (type == POSITION_TYPE_BUY) ? 1 : -1;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double fav = (side > 0) ? (bid - g_entry) : (g_entry - ask);
   double fav_r = fav / g_r;
   int max_r = MathMax(1, InpLadderMaxR);
   int k = (int)MathFloor(fav_r);
   if(k > max_r)
      k = max_r;
   if(k < 1)
      return;

   int want_rung = 0;
   double target = g_initial_sl;
   double be_off = MathMax(0.0, MathMin(InpBeOffsetR, 0.5)) * g_r;

   if(InpSLLadder)
     {
      want_rung = k;
      if(!InpSLBE && want_rung == 1)
         want_rung = 0;
      if(want_rung == 1)
         target = g_entry + (double)side * be_off;
      else if(want_rung >= 2)
         target = g_entry + (double)side * (want_rung - 1) * g_r;
     }
   else if(InpSLBE && k >= 1)
     {
      want_rung = 1;
      target = g_entry + (double)side * be_off;
     }

   if(want_rung <= g_rung)
      return;
   if(want_rung < 1)
      return;

   target = SB_NormPrice(target);
   string tag = (want_rung == 1) ? "SLBE" : StringFormat("SLLadder_%dR", want_rung);
   if(SB_ModifySL(g_pos_ticket, target, tag))
      g_rung = want_rung;
  }

void SB_CloseOurPositions(void)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      g_trade.PositionClose(ticket);
     }
   SB_ClearLadderState();
  }

bool SB_OpenSide(const int want)
  {
   SB_ClearLadderState();
   double lot = SB_TradeLot();
   string tag = (want > 0) ? "signals LONG" : "signals SHORT";
   if(StringLen(g_last_reason) > 0)
      tag = tag + " @" + g_last_reason;

   if(InpUseStopLoss && InpRequireStop)
     {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double entry = (want > 0) ? ask : bid;
      string serr;
      double sl = 0.0, r = 0.0;
      if(!SB_MakeSwingStop(want, entry, InpSwingLen, InpRMinAtr, InpRMaxAtr,
                           InpRMaxPct, InpChaseFrac, sl, r, serr, InpValidateR))
        {
         Print("Skip open (invalid R): ", serr);
         return false;
        }
     }

   bool ok = (want > 0) ? g_trade.Buy(lot, _Symbol, 0.0, 0.0, 0.0, tag)
                        : g_trade.Sell(lot, _Symbol, 0.0, 0.0, 0.0, tag);
   if(!ok || SB_CurrentPosDir() != want)
      return false;

   ulong ticket = SB_OurTicket();
   if(ticket != 0 && InpUseStopLoss)
     {
      if(!SB_AttachSwingSL(want, ticket) && InpRequireStop)
        {
         Print("Opened but swing SL failed — closing (RequireStop)");
         SB_CloseOurPositions();
         return false;
        }
     }
   return true;
  }

// AI-only: sync position to argmax (FLAT closes).
void SB_SyncAiOnly(const int cls)
  {
   int want = 0;
   if(cls == 1)
      want = 1;
   else if(cls == 2)
      want = -1;
   int have = SB_CurrentPosDir();
   if(want == have)
      return;
   if(have != 0)
      SB_CloseOurPositions();
   if(want == 0)
      return;
   SB_OpenSide(want);
  }

// Entry: LONG only if close>SMA && argmax LONG; SHORT only if close<SMA && argmax SHORT.
// FLAT never opens. Exit LONG when close<SMA; exit SHORT when close>SMA (AI ignored on exit).
void SB_SyncMaAiGate(const int cls)
  {
   int ai = 0;
   if(cls == 1)
      ai = 1;
   else if(cls == 2)
      ai = -1;

   double c = 0.0, fma = 0.0;
   string err;
   if(!SB_ClosedBarFastMa(_Symbol, InpFastMaLen, c, fma, err))
     {
      Print("MA gate: ", err);
      return;
     }

   int have = SB_CurrentPosDir();

   // MA exits (independent of AI)
   if(have > 0 && c < fma)
     {
      PrintFormat("MA exit LONG close=%.5f sma=%.5f", c, fma);
      SB_CloseOurPositions();
      have = 0;
     }
   else if(have < 0 && c > fma)
     {
      PrintFormat("MA exit SHORT close=%.5f sma=%.5f", c, fma);
      SB_CloseOurPositions();
      have = 0;
     }

   // Still in position: hold while MA agrees (even if AI goes FLAT/opposite)
   if(have != 0)
      return;

   // Flat: open only when AI side matches MA side (FLAT → no open)
   if(ai > 0 && c > fma)
     {
      PrintFormat("MA+AI entry LONG close=%.5f sma=%.5f", c, fma);
      SB_OpenSide(1);
     }
   else if(ai < 0 && c < fma)
     {
      PrintFormat("MA+AI entry SHORT close=%.5f sma=%.5f", c, fma);
      SB_OpenSide(-1);
     }
  }

// Chart parity: exit_arm (entry immediate; exit waits SMA14) + optional ST clip.
// Logical book tracks exit_arm; broker is flat when ST disagrees (re-enter when ST agrees).
void SB_SyncExitArmSt(const int cls)
  {
   int ai = 0;
   if(cls == 1)
      ai = 1;
   else if(cls == 2)
      ai = -1;

   double c = 0.0, fma = 0.0;
   string err = "";
   if(InpExitArm)
     {
      if(!SB_ClosedBarFastMa(_Symbol, InpFastMaLen, c, fma, err))
        {
         Print("exit_arm MA: ", err);
         return;
        }
     }

   int have = g_logical_have;

   // Exit when AI disagrees with open side (exit_arm waits for MA confirm).
   if(have != 0 && ai != have)
     {
      if(InpExitArm && !SB_ExitMaConfirm(have, c, fma))
        {
         // hold logical through AI FLAT/flip until MA confirms
        }
      else
        {
         PrintFormat("exit_arm EXIT %s ai=%s close=%.5f sma=%.5f",
                     (have > 0 ? "LONG" : "SHORT"), SB_ClassName(cls), c, fma);
         have = 0;
        }
     }

   // Flat logical book — enter immediately on AI side (entry_arm OFF).
   if(have == 0 && ai != 0)
     {
      have = ai;
      if(InpExitArm)
         PrintFormat("exit_arm ENTER %s close=%.5f sma=%.5f",
                     (have > 0 ? "LONG" : "SHORT"), c, fma);
      else
         PrintFormat("logical ENTER %s", (have > 0 ? "LONG" : "SHORT"));
     }

   g_logical_have = have;

   // ST filter: broker only holds when SuperTrend agrees (ATR15×3 like chart).
   int want = have;
   int st_dir = 0;
   if(InpStFilter && want != 0)
     {
      if(!SB_ClosedBarSuperTrendDir(_Symbol, InpStAtrLen, InpStFactor, st_dir, err))
        {
         Print("ST filter: ", err);
         return;
        }
      if((want > 0 && st_dir != 1) || (want < 0 && st_dir != -1))
        {
         g_last_reason = StringFormat("st_clip_%s", (st_dir > 0 ? "bull" : "bear"));
         want = 0;
        }
      else
         g_last_reason = StringFormat("st_ok_%s", (st_dir > 0 ? "bull" : "bear"));
     }

   int broker = SB_CurrentPosDir();
   if(want == broker)
      return;
   if(broker != 0)
      SB_CloseOurPositions();
   if(want == 0)
      return;
   SB_OpenSide(want);
  }

void SB_SyncTrade(const int cls)
  {
   if(!InpTradeRawAi)
      return;
   if(InpTradePolicy == SB_TRADE_SESSION)
      return; // handled in SB_SyncSessionTrade from OnTick
   // Chart parity path (default): exit_arm + ST filter.
   if(InpExitArm || InpStFilter)
     {
      SB_SyncExitArmSt(cls);
      return;
     }
   if(InpMaAiGate)
      SB_SyncMaAiGate(cls);
   else
      SB_SyncAiOnly(cls);
  }

// Session trade: decide on last H1 before Tokyo/London/NY; open at session start;
// hold until session end. Mid-session AI flips ignored. Parity with Python session_hold.
void SB_SyncSessionTrade(const int cls, const datetime closed_bar_server)
  {
   if(!InpTradeRawAi)
      return;

   datetime s0 = 0, e0 = 0, s1 = 0, e1 = 0;
   int sid = SB_SessionForBar(closed_bar_server, s0, e0);
   const bool decision_bar = SB_IsSessionDecisionBar(closed_bar_server);
   const bool last_bar = SB_IsSessionLastBar(closed_bar_server);

   // Close when the closed bar finishes a session window.
   if(last_bar && SB_CurrentPosDir() != 0)
     {
      PrintFormat("session_trade END %s → FLAT", SB_SessionName(sid));
      SB_CloseOurPositions();
     }

   // Off-hours and not a decide bar → flatten leftovers, clear lock.
   if(sid == SB_SESS_NONE && !decision_bar)
     {
      if(SB_CurrentPosDir() != 0)
        {
         PrintFormat("session_trade END off-hours → FLAT was=%s", SB_SessionName(g_sess_id));
         SB_CloseOurPositions();
        }
      g_sess_id = SB_SESS_NONE;
      g_sess_start_gmt = 0;
      g_sess_locked_cls = 0;
      g_sess_has_lock = false;
      g_last_reason = "off_session";
      return;
     }

   if(decision_bar)
     {
      datetime next = closed_bar_server + PeriodSeconds(PERIOD_H1);
      int nxt = SB_SessionForBar(next, s1, e1);
      g_sess_id = nxt;
      g_sess_start_gmt = s1;
      g_sess_locked_cls = cls;
      g_sess_has_lock = true;
      g_last_reason = StringFormat("decide_%s", SB_SessionName(nxt));
      PrintFormat("session_trade DECIDE %s lock=%s F=%.3f L=%.3f S=%.3f (open @ session start)",
                  SB_SessionName(nxt), SB_ClassName(cls),
                  g_last_probs[0], g_last_probs[1], g_last_probs[2]);
      return;
     }

   // Mid-session: hold locked side; do not re-enter if flat.
   if(g_sess_has_lock && sid == g_sess_id && s0 == g_sess_start_gmt)
     {
      g_last_reason = StringFormat("hold_%s", SB_SessionName(sid));
      int want = 0;
      if(g_sess_locked_cls == 1)
         want = 1;
      else if(g_sess_locked_cls == 2)
         want = -1;
      int have = SB_CurrentPosDir();
      if(want == 0)
        {
         if(have != 0)
            SB_CloseOurPositions();
         return;
        }
      if(have == want)
         return;
      if(have != 0)
         SB_CloseOurPositions();
      return;
     }

   // Mid-session without a lock → wait for next pre-session decide.
   if(SB_CurrentPosDir() != 0)
     {
      Print("session_trade: no lock mid-session → FLAT (wait next decide)");
      SB_CloseOurPositions();
     }
   if(!g_sess_has_lock)
     {
      g_sess_id = sid;
      g_sess_start_gmt = s0;
      g_sess_locked_cls = 0;
      g_last_reason = "wait_next_session";
     }
  }

// Open at the first forming H1 of a locked session (≈ session open fill).
void SB_SessionMaybeOpen(const datetime forming_bar_server)
  {
   if(!InpTradeRawAi || !g_sess_has_lock)
      return;
   if(!SB_IsSessionFirstBar(forming_bar_server))
      return;
   datetime s0 = 0, e0 = 0;
   int sid = SB_SessionForBar(forming_bar_server, s0, e0);
   if(sid != g_sess_id || s0 != g_sess_start_gmt)
      return;
   if(SB_CurrentPosDir() != 0)
      return; // already in (or wrong — SyncAiOnly will flip)
   PrintFormat("session_trade OPEN %s lock=%s",
               SB_SessionName(sid), SB_ClassName(g_sess_locked_cls));
   SB_SyncAiOnly(g_sess_locked_cls);
  }

int OnInit()
  {
   SB_SetSessionHoursMode((int)InpSessionHours);
   PrintFormat("Session hours: %s",
               InpSessionHours == SB_INP_SESS_ADJUSTED ? "adjusted" : "normal");
   g_asset = SB_AssetStemFromChart();
   g_logical_have = 0;
   g_trade_lot = SB_TradeLot();
   PrintFormat("Asset=%s chart=%s lot=%.4f (InpLot=%.4f; 0=auto XAU/ETH/XTI=0.1 XAG=0.5 BTC=0.01)",
               g_asset, _Symbol, g_trade_lot, InpLot);

   // session_trade = raw_ai only (argmax). Threshold / onset / MA / SL are ignored.
   if(InpTradePolicy == SB_TRADE_SESSION)
     {
      if(InpSignalMode != SB_MODE_RAW_AI)
         Print("WARN: session_trade forces raw_ai argmax (InpSignalMode ignored)");
      if(InpAiThreshold > 0.0)
         Print("WARN: session_trade ignores InpAiThreshold (raw_ai)");
      if(InpSignalPointGate || InpMaAiGate)
         Print("WARN: session_trade ignores signal/MA gates (raw_ai)");
      if(InpUseStopLoss)
         Print("WARN: session_trade recommended InpUseStopLoss=false (Python session_hold has no SL)");
      if(!InpTradeRawAi)
        {
         Print("FATAL: InpTradeRawAi must be true for session_trade backtest");
         return INIT_FAILED;
        }
      Print("Mode: session_trade + raw_ai only (argmax, no threshold/onset/MA)");
     }
   else
     {
      PrintFormat("Mode: bar + raw_ai | exit_arm=%s ST_filter=%s(ATR%d×%.1f) | pivot_breakout L=15",
                  (InpExitArm ? "ON" : "OFF"),
                  (InpStFilter ? "ON" : "OFF"),
                  InpStAtrLen, InpStFactor);
      if(!InpTradeRawAi)
        {
         Print("FATAL: InpTradeRawAi must be true for bar raw_ai backtest");
         return INIT_FAILED;
        }
      if(InpMaAiGate && (InpExitArm || InpStFilter))
         Print("WARN: InpMaAiGate ignored when exit_arm/ST filter enabled (chart parity)");
      if(InpUseStopLoss && (InpExitArm || InpStFilter))
         Print("WARN: chart parity path uses no SL — set InpUseStopLoss=false");
     }

   string parsed[];
   SB_ParseEngineList(InpEngines, parsed, g_n_engines);
   if(g_n_engines <= 0)
     {
      Print("InpEngines empty");
      return INIT_FAILED;
     }
   if(g_n_engines > SB_MAX_BLEND)
     {
      Print("Too many engines (max ", SB_MAX_BLEND, ")");
      return INIT_FAILED;
     }

   g_ready_n = 0;
   for(int i = 0; i < g_n_engines; i++)
     {
      g_engines[i] = parsed[i];
      int expect_ch = SB_EngineChannelCount(g_engines[i]);
      if(expect_ch <= 0)
        {
         Print("Unknown engine: ", g_engines[i]);
         return INIT_FAILED;
        }

      string onnx = SB_DefaultOnnxPath(g_asset, g_engines[i]);
      double temp = InpTemperature;
      int num_in = expect_ch;
      int lookback = InpLookback;
      SB_LoadSidecar(onnx, temp, num_in, lookback);
      if(InpTemperature > 1e-12)
         temp = InpTemperature;
      if(InpLookback > 0)
         lookback = InpLookback;
      if(num_in != expect_ch)
        {
         PrintFormat("WARN %s sidecar num_inputs=%d expected=%d — using expected",
                     g_engines[i], num_in, expect_ch);
         num_in = expect_ch;
        }

      g_models[i].SetTemperature(temp);
      g_models[i].SetLookback(lookback);
      g_models[i].SetNumInputs(num_in);
      if(g_models[i].Load(onnx))
         g_ready_n++;
      else
         Print("ONNX missing for ", g_engines[i], " @ ", onnx);
     }

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpSlippage);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   SB_ClearLadderState();

   ArrayInitialize(g_last_probs, 0.0);
   g_sess_id = SB_SESS_NONE;
   g_sess_start_gmt = 0;
   g_sess_locked_cls = 0;
   g_sess_has_lock = false;
   SB_UpdateComment();
   if(g_ready_n == 0)
     {
      Print("No ONNX loaded. Export engines then copy to MQL5/Files/", InpOnnxDir);
      Comment("SmartBS: waiting for ONNX under ", InpOnnxDir, g_asset, "_*.onnx");
     }
   else
      PrintFormat("Ready %d/%d for %s | policy=%s exit_arm=%s ST=%s mode=%s SL=%s",
                  g_ready_n, g_n_engines, g_asset,
                  (InpTradePolicy == SB_TRADE_SESSION ? "session_trade" : "bar"),
                  (InpExitArm ? "ON" : "OFF"),
                  (InpStFilter ? "ON" : "OFF"),
                  (InpSignalMode == SB_MODE_RAW_AI ? "raw_argmax" : "ai_threshold"),
                  (InpUseStopLoss ? "swing" : "OFF"));
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   Comment("");
   for(int i = 0; i < g_n_engines; i++)
      g_models[i].Shutdown();
  }

void OnTick()
  {
   if(g_ready_n <= 0)
      return;

   // Optional mechanical SL while in position.
   if(InpUseStopLoss)
     {
      SB_EnsureSwingSL();
      SB_UpdateSLBELadder();
     }

   datetime t = iTime(_Symbol, PERIOD_H1, 0);
   if(t == g_last_bar)
      return;
   if(InpOnlyClosedBar)
     {
      // new H1 forming -> previous just closed
     }
   g_last_bar = t;

   double sum_p[3];
   sum_p[0] = sum_p[1] = sum_p[2] = 0.0;
   int used = 0;
   string err;

   for(int i = 0; i < g_n_engines; i++)
     {
      if(!g_models[i].Ready())
         continue;
      float window[];
      int num_in = 0;
      if(!SB_BuildEngineLookbackWindow(g_engines[i], _Symbol,
                                       g_models[i].Lookback(), -1,
                                       window, num_in, err))
        {
         Print(g_engines[i], " features: ", err);
         continue;
        }
      if(num_in != g_models[i].NumInputs())
        {
         PrintFormat("%s ch mismatch feat=%d model=%d", g_engines[i], num_in, g_models[i].NumInputs());
         continue;
        }
      double probs[], logits[];
      int pred = g_models[i].Predict(window, probs, logits);
      if(pred < 0)
        {
         Print("FATAL: ", g_engines[i], " OnnxRun failed — fix model/features (not skipping quietly)");
         return;
        }
      sum_p[0] += probs[0];
      sum_p[1] += probs[1];
      sum_p[2] += probs[2];
      used++;
     }

   if(used <= 0)
      return;

   for(int k = 0; k < 3; k++)
      g_last_probs[k] = sum_p[k] / (double)used;
   double s = g_last_probs[0] + g_last_probs[1] + g_last_probs[2];
   if(s > 1e-12)
     {
      g_last_probs[0] /= s;
      g_last_probs[1] /= s;
      g_last_probs[2] /= s;
     }

   int cls = SB_Argmax(g_last_probs);
   // session_trade: always raw_ai argmax (ignore threshold / onset / MA).
   const bool session_raw = (InpTradePolicy == SB_TRADE_SESSION);
   if(!session_raw && InpSignalMode == SB_MODE_AI_THRESHOLD)
     {
      if(cls != 0 && g_last_probs[cls] < InpAiThreshold)
         cls = 0;
     }

   g_last_reason = "";
   // Bar-policy onset gate only (never under session_trade raw_ai).
   if(!session_raw && InpTradePolicy == SB_TRADE_BAR && InpSignalPointGate && !InpMaAiGate)
     {
      // Rebuild signals matrix and gate AI to engine-side onsets only.
      double sig_feats[][SB_SIGNALS_CH];
      int n_sig = 0;
      string serr = "";
      if(SB_BuildSignalsFeatureMatrix(_Symbol, sig_feats, n_sig, serr) && n_sig >= 2)
        {
         int end_bar = n_sig - 1;
         if(InpOnlyClosedBar && n_sig >= 3)
            end_bar = n_sig - 2; // last closed bar
         cls = SB_SignalsApplyOnsetGate(sig_feats, n_sig, end_bar, cls,
                                        InpSignalPointThr, InpSignalPointAgree,
                                        g_last_reason);
        }
      else if(StringLen(serr) > 0)
         Print("signal_point gate: ", serr);
     }

   g_last_cls = cls;
   SB_UpdateComment();

   datetime closed_bar = InpOnlyClosedBar ? iTime(_Symbol, PERIOD_H1, 1)
                                          : iTime(_Symbol, PERIOD_H1, 0);
   datetime forming_bar = iTime(_Symbol, PERIOD_H1, 0);
   if(session_raw)
     {
      PrintFormat("session_trade raw_ai=%s %s F=%.4f L=%.4f S=%.4f",
                  SB_ClassName(cls),
                  (StringLen(g_last_reason) > 0 ? g_last_reason : "-"),
                  g_last_probs[0], g_last_probs[1], g_last_probs[2]);
      SB_SyncSessionTrade(cls, closed_bar);
      SB_SessionMaybeOpen(forming_bar);
     }
   else
     {
      string gate = "raw_ai";
      if(InpExitArm || InpStFilter)
         gate = (InpExitArm && InpStFilter) ? "exit_arm+ST" :
                (InpExitArm ? "exit_arm" : "ST");
      else if(InpMaAiGate)
         gate = "ma_ai";
      else if(InpSignalPointGate)
         gate = "signal_point";
      PrintFormat("bar %s gate=%s %s logical=%d reason=%s F=%.4f L=%.4f S=%.4f",
                  (InpSignalMode == SB_MODE_RAW_AI ? "argmax" : "thr"),
                  gate,
                  SB_ClassName(cls),
                  g_logical_have,
                  (StringLen(g_last_reason) > 0 ? g_last_reason : "-"),
                  g_last_probs[0], g_last_probs[1], g_last_probs[2]);
      SB_SyncTrade(cls);
     }
  }
