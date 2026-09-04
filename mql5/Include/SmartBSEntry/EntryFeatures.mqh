//+------------------------------------------------------------------+
//| SmartBS Entry — 90-channel multi-TF features (engine.py parity)  |
//| Primary series: H1 chronological (oldest first).                 |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ENTRY_FEATURES_MQH
#define SMARTBS_ENTRY_FEATURES_MQH

#include "Indicators.mqh"
#include "Structure.mqh"

#define SB_MA_FAST     8
#define SB_MA_MID      21
#define SB_MA_SLOW     50
#define SB_MA_LONG     100
#define SB_RSI_PERIOD  14
#define SB_RSI_LOW     30.0
#define SB_RSI_HIGH    70.0
#define SB_SWING_LEN   5
#define SB_DECAY_CAP   24
#define SB_NUM_INPUTS  90
#define SB_PAIR_CH     15
#define SB_TF_PAIRS    6
#define SB_WARMUP_BARS (3 * SB_MA_LONG * 24)  // 7200

// Pair channel order must match Python PAIR_CHANNELS
// candle_strength, candle_dist, candle_movement, candle_dir,
// fast_dir, slow_dir, long_dir, regime_status,
// choch, bos, fgv,
// rsi_bull_reverse, rsi_bear_reverse, rsi_bull_entry, rsi_bear_entry

struct SBOhlc
  {
   datetime time[];
   double   open[];
   double   high[];
   double   low[];
   double   close[];
   int      n;
  };

bool SB_CopyRatesChrono(const string symbol, const ENUM_TIMEFRAMES tf, const int count, SBOhlc &out)
  {
   MqlRates rates[];
   int got = CopyRates(symbol, tf, 0, count, rates);
   if(got < 10)
      return false;
   // CopyRates: index 0 = newest → reverse to oldest-first
   out.n = got;
   ArrayResize(out.time, got);
   ArrayResize(out.open, got);
   ArrayResize(out.high, got);
   ArrayResize(out.low, got);
   ArrayResize(out.close, got);
   for(int i = 0; i < got; i++)
     {
      int j = got - 1 - i;
      out.time[i]  = rates[j].time;
      out.open[i]  = rates[j].open;
      out.high[i]  = rates[j].high;
      out.low[i]   = rates[j].low;
      out.close[i] = rates[j].close;
     }
   return true;
  }

void SB_MapLastLe(const datetime &dst_t[], const datetime &src_t[], const double &src_v[], double &out[])
  {
   int nd = ArraySize(dst_t);
   int ns = ArraySize(src_t);
   ArrayResize(out, nd);
   ArrayInitialize(out, 0.0);
   if(ns <= 0)
      return;
   int j = 0;
   for(int i = 0; i < nd; i++)
     {
      while(j + 1 < ns && src_t[j + 1] <= dst_t[i])
         j++;
      if(src_t[j] <= dst_t[i])
         out[i] = src_v[j];
      else
         out[i] = 0.0;
     }
   // ffill leading zeros once we have data
   int first = -1;
   for(int i = 0; i < nd; i++)
     {
      if(src_t[0] <= dst_t[i])
        {
         first = i;
         break;
        }
     }
   if(first > 0)
     {
      for(int i = 0; i < first; i++)
         out[i] = out[first];
     }
  }

void SB_CandleBlock(const double &o[], const double &h[], const double &l[], const double &c[],
                    double &strength[], double &dist[], double &movement[], double &dir[])
  {
   int n = ArraySize(c);
   ArrayResize(strength, n);
   ArrayResize(dist, n);
   ArrayResize(movement, n);
   ArrayResize(dir, n);
   double fast[], atr14[];
   SB_EMA(c, SB_MA_FAST, fast);
   SB_ATR(h, l, c, 14, atr14);
   for(int i = 0; i < n; i++)
     {
      double rng = h[i] - l[i];
      strength[i] = MathMax(-1.0, MathMin(1.0, SB_SafeDiv(o[i] - c[i], rng)));
      movement[i] = MathMax(-5.0, MathMin(5.0, SB_SafeDiv(o[i] - c[i], c[i]) * 100.0));
      dist[i] = MathMax(-5.0, MathMin(5.0, SB_SafeDiv(c[i] - fast[i], atr14[i])));
      double lo = MathMin(o[i], c[i]);
      double hi = MathMax(o[i], c[i]);
      if(fast[i] >= lo && fast[i] <= hi)
         dir[i] = 0.0;
      else
         dir[i] = (c[i] > fast[i] ? 1.0 : -1.0);
     }
  }

void SB_RegimeBlock(const double &c[], double &fast_dir[], double &slow_dir[],
                    double &long_dir[], double &regime_status[])
  {
   int n = ArraySize(c);
   double f[], m[], s[], lng[];
   SB_EMA(c, SB_MA_FAST, f);
   SB_EMA(c, SB_MA_MID, m);
   SB_EMA(c, SB_MA_SLOW, s);
   SB_EMA(c, SB_MA_LONG, lng);
   ArrayResize(fast_dir, n);
   ArrayResize(slow_dir, n);
   ArrayResize(long_dir, n);
   ArrayResize(regime_status, n);
   double spread[], prev_spread[];
   ArrayResize(spread, n);
   for(int i = 0; i < n; i++)
     {
      fast_dir[i] = (f[i] > s[i] ? 1.0 : -1.0);
      if(f[i] > s[i] && s[i] > m[i])
         slow_dir[i] = 1.0;
      else if(f[i] < s[i] && s[i] < m[i])
         slow_dir[i] = -1.0;
      else
         slow_dir[i] = 0.0;
      if(f[i] > s[i] && s[i] > m[i] && m[i] > lng[i])
         long_dir[i] = 1.0;
      else if(f[i] < s[i] && s[i] < m[i] && m[i] < lng[i])
         long_dir[i] = -1.0;
      else
         long_dir[i] = 0.0;
      double mx = MathMax(MathMax(f[i], m[i]), MathMax(s[i], lng[i]));
      double mn = MathMin(MathMin(f[i], m[i]), MathMin(s[i], lng[i]));
      spread[i] = mx - mn;
     }
   SB_PrevArray(spread, prev_spread);
   for(int i = 0; i < n; i++)
     {
      double den = MathMax(MathAbs(c[i]), 1e-9) * 0.001;
      regime_status[i] = MathTanh(SB_SafeDiv(spread[i] - prev_spread[i], den));
     }
  }

void SB_SmartMoneySigned(const double &h[], const double &l[], const double &c[],
                         double &choch[], double &bos[], double &fgv[])
  {
   int n = ArraySize(c);
   double atr14[];
   SB_ATR(h, l, c, 14, atr14);
   SBStructureBlock blk;
   SB_ComputeStructureBlock(h, l, c, atr14, SB_SWING_LEN, blk);
   ArrayResize(choch, n);
   ArrayResize(bos, n);
   ArrayResize(fgv, n);
   for(int i = 0; i < n; i++)
     {
      choch[i] = MathMax(-1.0, MathMin(1.0, blk.choch_up_decay[i] - blk.choch_dn_decay[i]));
      bos[i]   = MathMax(-1.0, MathMin(1.0, blk.bos_up_decay[i] - blk.bos_dn_decay[i]));
      fgv[i]   = MathMax(-1.0, MathMin(1.0, blk.fvg_prox_up[i] - blk.fvg_prox_dn[i]));
     }
  }

void SB_CrossDnFlags(const double &series[], const double level, bool &out[])
  {
   int n = ArraySize(series);
   ArrayResize(out, n);
   out[0] = false;
   for(int i = 1; i < n; i++)
      out[i] = (series[i - 1] >= level && series[i] < level);
  }

void SB_CrossUpFlags(const double &series[], const double level, bool &out[])
  {
   int n = ArraySize(series);
   ArrayResize(out, n);
   out[0] = false;
   for(int i = 1; i < n; i++)
      out[i] = (series[i - 1] <= level && series[i] > level);
  }

void SB_RsiPairBlock(const datetime &t1h[],
                     const datetime &t_maj[], const double &c_maj[],
                     const datetime &t_min[], const double &c_min[],
                     double &bull_rev[], double &bear_rev[],
                     double &bull_ent[], double &bear_ent[])
  {
   double rsi_maj[], rsi_min[];
   SB_RSI(c_maj, SB_RSI_PERIOD, rsi_maj);
   SB_RSI(c_min, SB_RSI_PERIOD, rsi_min);
   bool maj_dn[], min_dn[], maj_up[], min_up[];
   SB_CrossDnFlags(rsi_maj, SB_RSI_HIGH, maj_dn);
   SB_CrossDnFlags(rsi_min, SB_RSI_HIGH, min_dn);
   SB_CrossUpFlags(rsi_maj, SB_RSI_LOW, maj_up);
   SB_CrossUpFlags(rsi_min, SB_RSI_LOW, min_up);

   double d_maj_dn[], d_min_dn[], d_maj_up[], d_min_up[];
   SB_BarsSinceDecay(maj_dn, SB_DECAY_CAP, d_maj_dn);
   SB_BarsSinceDecay(min_dn, SB_DECAY_CAP, d_min_dn);
   SB_BarsSinceDecay(maj_up, SB_DECAY_CAP, d_maj_up);
   SB_BarsSinceDecay(min_up, SB_DECAY_CAP, d_min_up);

   double maj_dn70[], min_dn70[], maj_up30[], min_up30[], rsi_maj_1h[];
   SB_MapLastLe(t1h, t_maj, d_maj_dn, maj_dn70);
   SB_MapLastLe(t1h, t_min, d_min_dn, min_dn70);
   SB_MapLastLe(t1h, t_maj, d_maj_up, maj_up30);
   SB_MapLastLe(t1h, t_min, d_min_up, min_up30);
   SB_MapLastLe(t1h, t_maj, rsi_maj, rsi_maj_1h);

   int n = ArraySize(t1h);
   ArrayResize(bull_rev, n);
   ArrayResize(bear_rev, n);
   ArrayResize(bull_ent, n);
   ArrayResize(bear_ent, n);
   for(int i = 0; i < n; i++)
     {
      double maj_band = (rsi_maj_1h[i] > SB_RSI_LOW && rsi_maj_1h[i] < SB_RSI_HIGH) ? 1.0 : 0.0;
      bull_rev[i] = MathMax(0.0, MathMin(1.0, maj_dn70[i] * min_dn70[i]));
      bear_rev[i] = -MathMax(0.0, MathMin(1.0, maj_up30[i] * min_up30[i]));
      bull_ent[i] = MathMax(0.0, MathMin(1.0, maj_band * min_up30[i]));
      bear_ent[i] = -MathMax(0.0, MathMin(1.0, maj_band * min_dn70[i]));
     }
  }

void SB_PairFeaturesOnH1(const SBOhlc &h1, const SBOhlc &maj, const SBOhlc &minr, double &ch[][])
  {
   // ch[0..14][bar] — 15 channels on H1 length
   double strength[], dist[], movement[], dir[];
   SB_CandleBlock(minr.open, minr.high, minr.low, minr.close, strength, dist, movement, dir);
   double fast_dir[], slow_dir[], long_dir[], regime_status[];
   SB_RegimeBlock(maj.close, fast_dir, slow_dir, long_dir, regime_status);
   double choch[], bos[], fgv[];
   SB_SmartMoneySigned(minr.high, minr.low, minr.close, choch, bos, fgv);
   double bull_rev[], bear_rev[], bull_ent[], bear_ent[];
   SB_RsiPairBlock(h1.time, maj.time, maj.close, minr.time, minr.close,
                   bull_rev, bear_rev, bull_ent, bear_ent);

   double m_strength[], m_dist[], m_movement[], m_dir[];
   double m_choch[], m_bos[], m_fgv[];
   double m_fast[], m_slow[], m_long[], m_reg[];
   SB_MapLastLe(h1.time, minr.time, strength, m_strength);
   SB_MapLastLe(h1.time, minr.time, dist, m_dist);
   SB_MapLastLe(h1.time, minr.time, movement, m_movement);
   SB_MapLastLe(h1.time, minr.time, dir, m_dir);
   SB_MapLastLe(h1.time, minr.time, choch, m_choch);
   SB_MapLastLe(h1.time, minr.time, bos, m_bos);
   SB_MapLastLe(h1.time, minr.time, fgv, m_fgv);
   SB_MapLastLe(h1.time, maj.time, fast_dir, m_fast);
   SB_MapLastLe(h1.time, maj.time, slow_dir, m_slow);
   SB_MapLastLe(h1.time, maj.time, long_dir, m_long);
   SB_MapLastLe(h1.time, maj.time, regime_status, m_reg);

   int n = h1.n;
   ArrayResize(ch, 15);
   for(int k = 0; k < 15; k++)
      ArrayResize(ch[k], n);

   for(int i = 0; i < n; i++)
     {
      ch[0][i]  = m_strength[i];
      ch[1][i]  = m_dist[i];
      ch[2][i]  = m_movement[i];
      ch[3][i]  = m_dir[i];
      ch[4][i]  = m_fast[i];
      ch[5][i]  = m_slow[i];
      ch[6][i]  = m_long[i];
      ch[7][i]  = m_reg[i];
      ch[8][i]  = m_choch[i];
      ch[9][i]  = m_bos[i];
      ch[10][i] = m_fgv[i];
      ch[11][i] = bull_rev[i];
      ch[12][i] = bear_rev[i];
      ch[13][i] = bull_ent[i];
      ch[14][i] = bear_ent[i];
     }
  }

void SB_TfPairNames(const int pair_idx, string &major, string &minor)
  {
   switch(pair_idx)
     {
      case 0: major = "1w";  minor = "1d";  break;
      case 1: major = "1d";  minor = "4h";  break;
      case 2: major = "4h";  minor = "1h";  break;
      case 3: major = "1h";  minor = "15m"; break;
      case 4: major = "15m"; minor = "5m";  break;
      case 5: major = "5m";  minor = "1m";  break;
      default: major = "1h"; minor = "1h"; break;
     }
  }

bool SB_SelectTf(const string name, const SBOhlc &h1,
                 const SBOhlc &m1, const SBOhlc &m5, const SBOhlc &m15,
                 const SBOhlc &h4, const SBOhlc &d1, const SBOhlc &w1,
                 SBOhlc &out)
  {
   if(name == "1w" && w1.n > SB_MA_LONG) { out = w1; return true; }
   if(name == "1d" && d1.n > SB_MA_LONG) { out = d1; return true; }
   if(name == "4h" && h4.n > SB_MA_LONG) { out = h4; return true; }
   if(name == "1h") { out = h1; return true; }
   if(name == "15m" && m15.n > SB_MA_LONG) { out = m15; return true; }
   if(name == "5m" && m5.n > SB_MA_LONG) { out = m5; return true; }
   if(name == "1m" && m1.n > SB_MA_LONG) { out = m1; return true; }
   // Fallbacks matching Python engine
   if(name == "1m" && m5.n > SB_MA_LONG) { out = m5; return true; }
   if(name == "5m" && m15.n > SB_MA_LONG) { out = m15; return true; }
   out = h1;
   return true;
  }

// Build full (n_h1, 90) feature matrix. features[channel][bar], oldest-first.
// Returns false if history is insufficient.
bool SB_BuildFeatureMatrix(const string symbol, double &features[][], int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1;
   int need_h1 = SB_WARMUP_BARS + 128;
   if(!SB_CopyRatesChrono(symbol, PERIOD_H1, need_h1, h1))
     {
      err = StringFormat("H1 history insufficient (got %d, need ~%d)", h1.n, need_h1);
      return false;
     }
   n_bars = h1.n;

   SBOhlc cache_m1, cache_m5, cache_m15, cache_h4, cache_d1, cache_w1;
   int need_m1  = (int)MathMin(need_h1 * 60 + 500, 100000);
   int need_m5  = (int)MathMin(need_h1 * 12 + 200, 50000);
   int need_m15 = (int)MathMin(need_h1 * 4 + 200, 20000);
   int need_h4  = (int)MathMin(need_h1 / 4 + 200, 5000);
   int need_d1  = (int)MathMin(need_h1 / 24 + 100, 2000);
   int need_w1  = (int)MathMin(need_h1 / 120 + 50, 500);
   SB_CopyRatesChrono(symbol, PERIOD_M1, need_m1, cache_m1);
   SB_CopyRatesChrono(symbol, PERIOD_M5, need_m5, cache_m5);
   SB_CopyRatesChrono(symbol, PERIOD_M15, need_m15, cache_m15);
   SB_CopyRatesChrono(symbol, PERIOD_H4, need_h4, cache_h4);
   SB_CopyRatesChrono(symbol, PERIOD_D1, need_d1, cache_d1);
   SB_CopyRatesChrono(symbol, PERIOD_W1, need_w1, cache_w1);

   ArrayResize(features, SB_NUM_INPUTS);
   for(int c = 0; c < SB_NUM_INPUTS; c++)
     {
      ArrayResize(features[c], n_bars);
      ArrayInitialize(features[c], 0.0);
     }

   for(int p = 0; p < SB_TF_PAIRS; p++)
     {
      string maj_n, min_n;
      SB_TfPairNames(p, maj_n, min_n);
      SBOhlc maj, minr;
      SB_SelectTf(maj_n, h1, cache_m1, cache_m5, cache_m15, cache_h4, cache_d1, cache_w1, maj);
      SB_SelectTf(min_n, h1, cache_m1, cache_m5, cache_m15, cache_h4, cache_d1, cache_w1, minr);
      if(maj.n < SB_MA_LONG + 5)
         maj = h1;
      if(minr.n < SB_MA_LONG + 5)
         minr = h1;

      double ch[][];
      SB_PairFeaturesOnH1(h1, maj, minr, ch);
      int base = p * SB_PAIR_CH;
      for(int k = 0; k < SB_PAIR_CH; k++)
        {
         for(int i = 0; i < n_bars; i++)
            features[base + k][i] = ch[k][i];
        }
     }
   return true;
  }

// Fill ONNX input buffer: layout [1, 90, lookback] = channel-major then time
// features_chw[c * lookback + t]  OR flat array of size 90*lookback
bool SB_BuildLookbackWindow(const double &features[][], const int n_bars,
                            const int lookback, const int end_bar,
                            float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_NUM_INPUTS * lookback);
   for(int c = 0; c < SB_NUM_INPUTS; c++)
     {
      for(int t = 0; t < lookback; t++)
        {
         int bar = end_bar - lookback + 1 + t;
         window[c * lookback + t] = (float)features[c][bar];
        }
     }
   return true;
  }

#endif
