//+------------------------------------------------------------------+
//| SmartBS — common channels appended to every engine window        |
//| Parity with smartbs_engines/common_channels.py (16 channels)     |
//+------------------------------------------------------------------+
#ifndef SMARTBS_COMMON_CHANNELS_MQH
#define SMARTBS_COMMON_CHANNELS_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"

#define SB_COMMON_CH           16
#define SB_COMMON_EMA_FAST     14
#define SB_COMMON_EMA_MID      48
#define SB_COMMON_EMA_SLOW     120
#define SB_COMMON_VOL_SMA      48
#define SB_COMMON_VOL_CLIP     3.0
#define SB_COMMON_WARMUP       (3 * SB_COMMON_EMA_SLOW)
#define SB_COMMON_DAY_SEC      86400
#define SB_CLOSE_DISP_CLIP     0.05

// UTC session ids (must be above SB_BuildCommonFeatureMatrix)
#define SB_SESS_NONE    0
#define SB_SESS_TOKYO   1
#define SB_SESS_LONDON  2
#define SB_SESS_NY      3

// Session hour packs (parity with common_channels.py). Default = normal.
#define SB_SESS_HOURS_NORMAL    0  // Tokyo 0-9, London 7-16, NY 12-21
#define SB_SESS_HOURS_ADJUSTED  1  // Tokyo 0-7, London 7-12, NY 12-20

int g_sb_session_hours_mode = SB_SESS_HOURS_NORMAL;

void SB_SetSessionHoursMode(const int mode)
  {
   g_sb_session_hours_mode =
      (mode == SB_SESS_HOURS_ADJUSTED) ? SB_SESS_HOURS_ADJUSTED : SB_SESS_HOURS_NORMAL;
  }

int SB_GetSessionHoursMode()
  {
   return g_sb_session_hours_mode;
  }

void SB_SessionBounds(const int sid, int &lo, int &hi)
  {
   const bool adj = (g_sb_session_hours_mode == SB_SESS_HOURS_ADJUSTED);
   if(sid == SB_SESS_TOKYO)
     {
      lo = 0;
      hi = adj ? 7 : 9;
      return;
     }
   if(sid == SB_SESS_LONDON)
     {
      lo = 7;
      hi = adj ? 12 : 16;
      return;
     }
   if(sid == SB_SESS_NY)
     {
      lo = 12;
      hi = adj ? 20 : 21;
      return;
     }
   lo = 0;
   hi = 0;
  }

// Indices match COMMON_FEATURE_NAMES in common_channels.py
// 0..3  ohlcv_open/high/low/volume
// 4     ohlcv_close_displacement  (Δclose/prev_close, clip ±5%)
// 5..6  tod_sin, tod_cos
// 7..9  session_tokyo, session_london, session_newyork
// 10    cd_vs_fast (−1/0/+1)
// 11    cd_body_cross_fast
// 12    session_strength (final of last completed session)
// 13    current_session_strength (running sum in forming session)
// 14    trend_regime (+1 bull 3MA EMA14>48>120 / −1 bear / 0)
// 15    last_candle_strength (open−close)/(high−low)

datetime SB_BarTimeGMT(const datetime bar_server_time);
int SB_SessionForBarGMT(const datetime bar_gmt,
                        datetime &sess_start_gmt,
                        datetime &sess_end_gmt);

bool SB_BuildCommonFeatureMatrix(const string symbol, double &features[][SB_COMMON_CH],
                                 int &n_bars, string &err)
  {
   err = "";
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int want = MathMax(SB_COMMON_WARMUP + 512, 400);
   // Prefer enough history for lookback-64 + warmup even early in tester.
   want = MathMax(want, SB_COMMON_WARMUP + 128);
   int got = CopyRates(symbol, SB_PrimaryTf(), 0, want, rates);
   if(got < 10)
     {
      err = "common: not enough H1 bars";
      return false;
     }
   n_bars = got;
   ArrayResize(features, n_bars);
   ArrayInitialize(features, 0.0);

   double open[], high[], low[], close[], volume[];
   ArrayResize(open, n_bars);
   ArrayResize(high, n_bars);
   ArrayResize(low, n_bars);
   ArrayResize(close, n_bars);
   ArrayResize(volume, n_bars);
   datetime times[];
   ArrayResize(times, n_bars);
   for(int i = 0; i < n_bars; i++)
     {
      times[i] = rates[i].time;
      open[i]  = rates[i].open;
      high[i]  = rates[i].high;
      low[i]   = rates[i].low;
      close[i] = rates[i].close;
      volume[i] = (double)rates[i].tick_volume;
     }

   double ema_f[], ema_mid[], ema_slow[], vol_sma[];
   SB_EMA(close, SB_COMMON_EMA_FAST, ema_f);
   SB_EMA(close, SB_COMMON_EMA_MID, ema_mid);
   SB_EMA(close, SB_COMMON_EMA_SLOW, ema_slow);
   SB_SMA(volume, SB_COMMON_VOL_SMA, vol_sma);

   double sess_run = 0.0;
   double sess_completed = 0.0;
   datetime prev_sess_start = 0;
   bool have_sess = false;
   bool have_completed = false;

   for(int i = 0; i < n_bars; i++)
     {
      double c = close[i];
      features[i][0] = SB_Nan0(SB_SafeDiv(open[i], c));
      features[i][1] = SB_Nan0(SB_SafeDiv(high[i], c));
      features[i][2] = SB_Nan0(SB_SafeDiv(low[i], c));
      features[i][3] = SB_Clip(SB_Nan0(SB_SafeDiv(volume[i], vol_sma[i])), 0.0, SB_COMMON_VOL_CLIP);
      if(i > 0)
         features[i][4] = SB_Clip(SB_Nan0(SB_SafeDiv(c - close[i - 1], close[i - 1])),
                                  -SB_CLOSE_DISP_CLIP, SB_CLOSE_DISP_CLIP);
      else
         features[i][4] = 0.0;

      datetime bar_gmt = SB_BarTimeGMT(times[i]);
      int sec = (int)(bar_gmt % SB_COMMON_DAY_SEC);
      if(sec < 0)
         sec += SB_COMMON_DAY_SEC;
      double frac = (double)sec / (double)SB_COMMON_DAY_SEC;
      double ang = 2.0 * M_PI * frac;
      features[i][5] = MathSin(ang);
      features[i][6] = MathCos(ang);

      int hour = sec / 3600;
      // Session flags (parity with common_channels _session_cols; may overlap in normal)
      int t_lo = 0, t_hi = 0, l_lo = 0, l_hi = 0, n_lo = 0, n_hi = 0;
      SB_SessionBounds(SB_SESS_TOKYO, t_lo, t_hi);
      SB_SessionBounds(SB_SESS_LONDON, l_lo, l_hi);
      SB_SessionBounds(SB_SESS_NY, n_lo, n_hi);
      features[i][7] = (hour >= t_lo && hour < t_hi) ? 1.0 : 0.0;
      features[i][8] = (hour >= l_lo && hour < l_hi) ? 1.0 : 0.0;
      features[i][9] = (hour >= n_lo && hour < n_hi) ? 1.0 : 0.0;

      if(close[i] > ema_f[i])
         features[i][10] = 1.0;
      else if(close[i] < ema_f[i])
         features[i][10] = -1.0;
      else
         features[i][10] = 0.0;
      double body_lo = MathMin(open[i], close[i]);
      double body_hi = MathMax(open[i], close[i]);
      bool body_cross = (body_lo < ema_f[i] && body_hi > ema_f[i]);
      features[i][11] = body_cross ? 1.0 : 0.0;

      // session_strength / current_session_strength (parity with _session_strength_pair)
      datetime s0 = 0, e0 = 0;
      int sid = SB_SessionForBarGMT(bar_gmt, s0, e0);
      double bar_score = 0.0;
      if(body_cross)
         bar_score = 0.0;
      else if(close[i] > ema_f[i])
         bar_score = 1.0;
      else if(close[i] < ema_f[i])
         bar_score = -1.0;
      features[i][12] = have_completed ? sess_completed : 0.0;
      if(sid == SB_SESS_NONE)
        {
         sess_run = 0.0;
         have_sess = false;
         features[i][13] = 0.0;
        }
      else
        {
         if(!have_sess || s0 != prev_sess_start)
           {
            sess_run = 0.0;
            prev_sess_start = s0;
            have_sess = true;
           }
         sess_run += bar_score;
         features[i][13] = sess_run;
         // Last bar of session?
         bool last_of_sess = true;
         if(i + 1 < n_bars)
           {
            datetime s1 = 0, e1 = 0;
            datetime next_gmt = SB_BarTimeGMT(times[i + 1]);
            int sid1 = SB_SessionForBarGMT(next_gmt, s1, e1);
            if(sid1 == sid && s1 == s0)
               last_of_sess = false;
           }
         if(last_of_sess)
           {
            sess_completed = sess_run;
            have_completed = true;
           }
        }

      // trend_regime: +1 bull 3MA stack (EMA14>48>120), −1 bear, else 0
      if(ema_f[i] > ema_mid[i] && ema_mid[i] > ema_slow[i])
         features[i][14] = 1.0;
      else if(ema_f[i] < ema_mid[i] && ema_mid[i] < ema_slow[i])
         features[i][14] = -1.0;
      else
         features[i][14] = 0.0;

      // last_candle_strength = (open − close) / (high − low)
      features[i][15] = SB_Nan0(SB_SafeDiv(open[i] - close[i], high[i] - low[i]));
     }
   return true;
  }

// --- UTC session clock (parity with common_channels session_start_end_ms) ---
// Normal (default, overlapping): Tokyo 0-9, London 7-16, NY 12-21
// Adjusted (opt-in):             Tokyo 0-7, London 7-12, NY 12-20
// When overlapping, pick the session with the latest end (matches Python).
// Off-hours: after NY end until Tokyo open.

datetime SB_BarTimeGMT(const datetime bar_server_time)
  {
   // Convert chart/server bar time → GMT using current server↔GMT offset.
   return bar_server_time - TimeCurrent() + TimeGMT();
  }

int SB_UtcHourOfBar(const datetime bar_server_time)
  {
   datetime gmt = SB_BarTimeGMT(bar_server_time);
   MqlDateTime dt;
   TimeToStruct(gmt, dt);
   return dt.hour;
  }

// Returns session id for the bar; also session [start,end) in GMT seconds.
int SB_SessionForBarGMT(const datetime bar_gmt,
                        datetime &sess_start_gmt,
                        datetime &sess_end_gmt)
  {
   sess_start_gmt = 0;
   sess_end_gmt = 0;
   MqlDateTime dt;
   TimeToStruct(bar_gmt, dt);
   datetime day0 = bar_gmt - (datetime)(dt.hour * 3600 + dt.min * 60 + dt.sec);
   const int hour = dt.hour;

   int best_sid = SB_SESS_NONE;
   int best_lo = 0, best_hi = -1;
   for(int sid = SB_SESS_TOKYO; sid <= SB_SESS_NY; sid++)
     {
      int lo = 0, hi = 0;
      SB_SessionBounds(sid, lo, hi);
      if(hour >= lo && hour < hi && hi > best_hi)
        {
         best_sid = sid;
         best_lo = lo;
         best_hi = hi;
        }
     }
   if(best_sid == SB_SESS_NONE)
      return SB_SESS_NONE;
   sess_start_gmt = day0 + (datetime)(best_lo * 3600);
   sess_end_gmt = day0 + (datetime)(best_hi * 3600);
   return best_sid;
  }

int SB_SessionForBar(const datetime bar_server_time,
                     datetime &sess_start_gmt,
                     datetime &sess_end_gmt)
  {
   return SB_SessionForBarGMT(SB_BarTimeGMT(bar_server_time), sess_start_gmt, sess_end_gmt);
  }

string SB_SessionName(const int sid)
  {
   if(sid == SB_SESS_TOKYO)
      return "Tokyo";
   if(sid == SB_SESS_LONDON)
      return "London";
   if(sid == SB_SESS_NY)
      return "NY";
   return "None";
  }

// First owned bar of this session? Compare to previous H1 bar ownership.
bool SB_IsSessionFirstBar(const datetime bar_server_time)
  {
   datetime s0 = 0, e0 = 0, s1 = 0, e1 = 0;
   int cur = SB_SessionForBar(bar_server_time, s0, e0);
   if(cur == SB_SESS_NONE)
      return false;
   datetime prev = bar_server_time - PeriodSeconds(SB_PrimaryTf());
   int prev_sid = SB_SessionForBar(prev, s1, e1);
   if(prev_sid != cur || s1 != s0)
      return true;
   return false;
  }

// Pre-session decision bar (parity with Python session_hold / session_trend):
// last H1 before an upcoming session starts (next hour enters a new [start,end)).
bool SB_IsSessionDecisionBar(const datetime bar_server_time)
  {
   datetime s0 = 0, e0 = 0, s1 = 0, e1 = 0;
   int cur = SB_SessionForBar(bar_server_time, s0, e0);
   datetime next = bar_server_time + PeriodSeconds(SB_PrimaryTf());
   int nxt = SB_SessionForBar(next, s1, e1);
   if(nxt == SB_SESS_NONE)
      return false;
   if(cur == SB_SESS_NONE)
      return true;
   return (s1 != s0);
  }

// Last H1 inside a session (next hour leaves this session window).
bool SB_IsSessionLastBar(const datetime bar_server_time)
  {
   datetime s0 = 0, e0 = 0, s1 = 0, e1 = 0;
   int cur = SB_SessionForBar(bar_server_time, s0, e0);
   if(cur == SB_SESS_NONE)
      return false;
   datetime next = bar_server_time + PeriodSeconds(SB_PrimaryTf());
   int nxt = SB_SessionForBar(next, s1, e1);
   if(nxt == SB_SESS_NONE)
      return true;
   return (s1 != s0);
  }

// Build common-only lookback window (channel-major, 16 × lookback).
bool SB_BuildCommonLookbackWindow(const string symbol,
                                  const int lookback,
                                  float &window[],
                                  string &err)
  {
   err = "";
   if(lookback <= 0)
     {
      err = "common: invalid lookback";
      return false;
     }
   double common[][SB_COMMON_CH];
   int n_bars = 0;
   if(!SB_BuildCommonFeatureMatrix(symbol, common, n_bars, err))
      return false;
   const int end_bar = n_bars - 2;
   if(end_bar < lookback - 1 || end_bar < 0)
     {
      err = StringFormat("common: need lookback=%d closed bars, have n=%d", lookback, n_bars);
      return false;
     }
   ArrayResize(window, SB_COMMON_CH * lookback);
   for(int c = 0; c < SB_COMMON_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] =
            (float)common[end_bar - lookback + 1 + t][c];
   return true;
  }

// Append common channels onto an existing core lookback window
// (channel-major layout: core_ch blocks, then SB_COMMON_CH blocks).
//
// ``end_bar`` from the engine matrix is NOT reused as an absolute index into
// common: engines often load more H1 bars than common. Both buffers are
// "most recent N bars", so we always slice common at its last closed bar
// (n_bars-2), matching engine ``end_bar = n_bars-2`` semantics.
bool SB_ExtendWindowWithCommon(const string symbol,
                               const int lookback,
                               const int /*end_bar_from_engine*/,
                               const int core_ch,
                               float &window[],
                               string &err)
  {
   err = "";
   if(core_ch <= 0 || lookback <= 0)
     {
      err = "common: invalid core_ch/lookback";
      return false;
     }
   double common[][SB_COMMON_CH];
   int n_bars = 0;
   if(!SB_BuildCommonFeatureMatrix(symbol, common, n_bars, err))
      return false;
   // Last closed H1 in this buffer (forming bar = n_bars-1).
   const int end_bar = n_bars - 2;
   if(end_bar < lookback - 1 || end_bar < 0)
     {
      err = StringFormat("common: need lookback=%d closed bars, have n=%d", lookback, n_bars);
      return false;
     }

   ArrayResize(window, (core_ch + SB_COMMON_CH) * lookback);
   for(int c = 0; c < SB_COMMON_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[(core_ch + c) * lookback + t] =
            (float)common[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
