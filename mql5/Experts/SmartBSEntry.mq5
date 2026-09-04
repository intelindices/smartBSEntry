//+------------------------------------------------------------------+
//| SmartBSEntry.mq5 — Entry engine + ONNX TCN raw_ai Expert Advisor |
//| Chart: H1. Signal: argmax FLAT/LONG/SHORT. Lot: fixed.           |
//+------------------------------------------------------------------+
#property copyright "SmartBS Entry"
#property version   "0.9.7"
#property strict

#include <Trade\Trade.mqh>
#include <SmartBSEntry\OnnxModel.mqh>
#include <SmartBSEntry\EntryFeatures.mqh>
#include <SmartBSEntry\Softmax.mqh>

input string InpOnnxFile     = "SmartBSEntry\\XAUUSD.onnx"; // ONNX under MQL5/Files/
input double InpTemperature  = 1.0;      // Softmax temperature (from sidecar JSON)
input int    InpLookback     = 64;       // TCN lookback
input double InpLot          = 0.1;      // Fixed lot size
input ulong  InpMagic        = 20260904;
input int    InpSlippage     = 30;
input bool   InpOnlyClosedBar = true;    // Trade only on new H1 bar
input bool   InpAllowTrade   = true;

CSBOnnxModel g_model;
CTrade       g_trade;
datetime     g_last_bar = 0;

int OnInit()
  {
   if(_Period != PERIOD_H1)
      Print("WARNING: SmartBS Entry was trained on H1 — attach to H1 chart.");

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpSlippage);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   g_model.SetTemperature(InpTemperature);
   g_model.SetLookback(InpLookback);
   if(!g_model.Load(InpOnnxFile))
     {
      Print("Failed to load ONNX: ", InpOnnxFile);
      Print("Copy .onnx to MQL5/Files/SmartBSEntry/ (or Common\\Files).");
      return INIT_FAILED;
     }
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   g_model.Shutdown();
  }

int CurrentSide()
  {
   // 0=flat, 1=long, -1=short (our magic only)
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
         return 1;
      if(type == POSITION_TYPE_SELL)
         return -1;
     }
   return 0;
  }

bool CloseAllOurs()
  {
   bool ok = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      if(!g_trade.PositionClose(ticket))
        {
         Print("Close failed ticket=", ticket, " err=", GetLastError());
         ok = false;
        }
     }
   return ok;
  }

void ApplySignal(const int cls)
  {
   // 0=FLAT 1=LONG 2=SHORT
   int side = CurrentSide();
   int want = 0;
   if(cls == 1)
      want = 1;
   else if(cls == 2)
      want = -1;

   if(want == 0)
     {
      if(side != 0)
         CloseAllOurs();
      return;
     }
   if(side == want)
      return; // already on the book
   if(side != 0)
      CloseAllOurs();
   if(!InpAllowTrade)
      return;
   if(want > 0)
      g_trade.Buy(InpLot, _Symbol);
   else
      g_trade.Sell(InpLot, _Symbol);
  }

void OnTick()
  {
   if(!g_model.Ready())
      return;

   datetime t[];
   if(CopyTime(_Symbol, PERIOD_H1, 0, 2, t) < 2)
      return;
   // t[0]=current forming, t[1]=last closed — use closed bar when OnlyClosedBar
   datetime closed = t[1];
   if(InpOnlyClosedBar)
     {
      if(closed == g_last_bar)
         return;
     }

   string err;
   double features[][];
   int n_bars = 0;
   if(!SB_BuildFeatureMatrix(_Symbol, features, n_bars, err))
     {
      Print("Features: ", err);
      g_last_bar = closed; // avoid spam every tick
      return;
     }
   if(n_bars < SB_WARMUP_BARS + InpLookback)
     {
      PrintFormat("Warmup incomplete: bars=%d need>=%d", n_bars, SB_WARMUP_BARS + InpLookback);
      g_last_bar = closed;
      return;
     }

   // Last closed H1 bar in chrono array = n_bars-2 if current forming is included,
   // or n_bars-1 if CopyRates starts at forming bar 0 (newest). We reversed to oldest-first,
   // so newest = n_bars-1 (forming). Closed = n_bars-2.
   int end_bar = n_bars - 2;
   if(end_bar < InpLookback - 1)
      end_bar = n_bars - 1;

   float window[];
   if(!SB_BuildLookbackWindow(features, n_bars, InpLookback, end_bar, window, err))
     {
      Print("Window: ", err);
      return;
     }

   double probs[], logits[];
   int cls = g_model.Predict(window, probs, logits);
   string names[3] = {"FLAT", "LONG", "SHORT"};
   PrintFormat("SmartBS raw_ai %s conf=%.3f probs=[%.3f %.3f %.3f] bar=%d",
               names[cls], probs[cls], probs[0], probs[1], probs[2], end_bar);

   ApplySignal(cls);
   g_last_bar = closed;
  }
