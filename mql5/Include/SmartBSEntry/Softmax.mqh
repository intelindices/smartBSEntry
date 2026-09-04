//+------------------------------------------------------------------+
//| Softmax with temperature (calibration.py parity)                 |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ENTRY_SOFTMAX_MQH
#define SMARTBS_ENTRY_SOFTMAX_MQH

void SB_SoftmaxTemp(const double &logits[], const double temperature, double &probs[])
  {
   int n = ArraySize(logits);
   ArrayResize(probs, n);
   if(n <= 0)
      return;
   double t = (temperature > 1e-8 ? temperature : 1.0);
   double mx = logits[0];
   for(int i = 1; i < n; i++)
      if(logits[i] > mx)
         mx = logits[i];
   double sum = 0.0;
   for(int i = 0; i < n; i++)
     {
      probs[i] = MathExp((logits[i] - mx) / t);
      sum += probs[i];
     }
   if(sum <= 0.0)
      sum = 1.0;
   for(int i = 0; i < n; i++)
      probs[i] /= sum;
  }

int SB_Argmax(const double &v[])
  {
   int n = ArraySize(v);
   if(n <= 0)
      return 0;
   int best = 0;
   for(int i = 1; i < n; i++)
      if(v[i] > v[best])
         best = i;
   return best;
  }

#endif
