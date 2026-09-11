//+------------------------------------------------------------------+
//| SmartBSEntry.mq5 — ONNX raw_ai signal shell (no trading)         |
//| Chart: H1. Default features: dBB 26ch (1h + last-completed 15m). |
//| Bake ONNX later via smartbs_engines.export_onnx; place under     |
//| MQL5/Files/SmartBSEntry/ or set InpOnnxFile.                     |
//+------------------------------------------------------------------+
#property copyright "SmartBS"
#property version   "2.00"
#property strict
#property description "ONNX → raw_ai only. Risk / orders come later."

#include <SmartBSEntry\OnnxModel.mqh>
#include <SmartBSEntry\DbbFeatures.mqh>
#include <SmartBSEntry\Softmax.mqh>

enum ENUM_SB_SIGNAL_MODE
  {
   SB_MODE_RAW_AI       = 0, // argmax only
   SB_MODE_AI_THRESHOLD = 1  // argmax + min conf → else FLAT
  };

input string              InpOnnxFile     = "";        // blank = SmartBSEntry\{ASSET}_dbb.onnx
input string              InpEngines      = "dbb";     // comma list (only dbb features ported yet)
input double              InpTemperature  = 1.0;       // override; 0 = use sidecar if present
input ENUM_SB_SIGNAL_MODE InpSignalMode   = SB_MODE_RAW_AI;
input double              InpAiThreshold  = 0.35;      // conf < th → FLAT (threshold mode)
input int                 InpLookback     = 64;
input int                 InpNumInputs    = 0;         // 0 = from sidecar / dBB default 26
input bool                InpOnlyClosedBar = true;

CSBOnnxModel g_model;
datetime     g_last_bar = 0;
string       g_asset = "XAUUSD";
string       g_engine = "dbb";
double       g_temperature = 1.0;
int          g_num_inputs = SB_NUM_INPUTS;
int          g_last_cls = 0;
double       g_last_probs[3];

string SB_AssetStemFromChart(void)
  {
   string s = _Symbol;
   StringToUpper(s);
   if(StringFind(s, "XAU") >= 0 || StringFind(s, "GOLD") >= 0)
      return "XAUUSD";
   if(StringFind(s, "XAG") >= 0 || StringFind(s, "SILVER") >= 0)
      return "XAGUSD";
   int dot = StringFind(s, ".");
   if(dot > 0)
      s = StringSubstr(s, 0, dot);
   return s;
  }

string SB_FirstEngine(const string engines)
  {
   string e = engines;
   StringTrimLeft(e);
   StringTrimRight(e);
   int comma = StringFind(e, ",");
   if(comma > 0)
      e = StringSubstr(e, 0, comma);
   StringToLower(e);
   return e;
  }

string SB_DefaultOnnxPath(const string asset, const string engine)
  {
   return "SmartBSEntry\\" + asset + "_" + engine + ".onnx";
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

   // Minimal key scrape (sidecar is small / flat JSON from export_onnx).
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
   string line = StringFormat(
      "SmartBS raw_ai | %s | eng=%s | %s | F=%.3f L=%.3f S=%.3f",
      g_asset, g_engine, SB_ClassName(g_last_cls),
      g_last_probs[0], g_last_probs[1], g_last_probs[2]);
   Comment(line);
  }

int OnInit()
  {
   g_asset = SB_AssetStemFromChart();
   g_engine = SB_FirstEngine(InpEngines);
   if(g_engine != "dbb")
     {
      Print("Only feature builder ported today: dbb. Got eng=", g_engine,
            " — set InpEngines=dbb until other MQL ports exist.");
      return INIT_FAILED;
     }

   string onnx = InpOnnxFile;
   if(onnx == "")
      onnx = SB_DefaultOnnxPath(g_asset, g_engine);

   double temp = InpTemperature;
   int num_in = (InpNumInputs > 0 ? InpNumInputs : SB_NUM_INPUTS);
   int lookback = InpLookback;
   SB_LoadSidecar(onnx, temp, num_in, lookback);
   if(InpTemperature > 1e-12)
      temp = InpTemperature;
   if(InpNumInputs > 0)
      num_in = InpNumInputs;
   if(InpLookback > 0)
      lookback = InpLookback;

   g_temperature = temp;
   g_num_inputs = num_in;
   g_model.SetTemperature(g_temperature);
   g_model.SetLookback(lookback);
   g_model.SetNumInputs(g_num_inputs);

   if(!g_model.Load(onnx))
     {
      Print("ONNX not loaded (", onnx, "). Train + export_onnx, then copy .onnx/.json under MQL5/Files/SmartBSEntry/. EA stays idle until then.");
      // Still init OK so chart comment can explain missing model.
      Comment("SmartBS: waiting for ONNX at ", onnx);
      return INIT_SUCCEEDED;
     }

   ArrayInitialize(g_last_probs, 0.0);
   SB_UpdateComment();
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   Comment("");
   g_model.Shutdown();
  }

void OnTick()
  {
   if(!g_model.Ready())
      return;

   datetime t = iTime(_Symbol, PERIOD_H1, 0);
   if(InpOnlyClosedBar)
     {
      if(t == g_last_bar)
         return;
      // Act on the bar that just closed (shift 1).
     }
   else if(t == g_last_bar)
      return;
   g_last_bar = t;

   string err;
   double features[][SB_NUM_INPUTS];
   int n_bars = 0;
   if(!SB_BuildDbbFeatureMatrix(_Symbol, features, n_bars, err))
     {
      Print("features: ", err);
      return;
     }

   int end_bar = n_bars - 2; // last closed H1
   if(end_bar < g_model.Lookback() - 1)
      return;

   float window[];
   if(!SB_BuildDbbLookbackWindow(features, n_bars, g_model.Lookback(), end_bar, window, err))
     {
      Print("window: ", err);
      return;
     }

   double probs[], logits[];
   int cls = g_model.Predict(window, probs, logits);
   if(InpSignalMode == SB_MODE_AI_THRESHOLD)
     {
      double conf = probs[cls];
      if(cls != 0 && conf < InpAiThreshold)
         cls = 0;
     }

   g_last_cls = cls;
   for(int i = 0; i < 3; i++)
      g_last_probs[i] = probs[i];
   SB_UpdateComment();
   PrintFormat("raw_ai %s F=%.4f L=%.4f S=%.4f",
               SB_ClassName(cls), probs[0], probs[1], probs[2]);
  }
