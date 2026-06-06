//+------------------------------------------------------------------+
//|                                                   TradingBot.mq5 |
//|                                  Copyright 2026, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Antigravity AI"
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade\Trade.mqh>

enum ENUM_SIGNAL_MODE
  {
   SIGNAL_WITH_BOOSTER = 0,
   SIGNAL_BASE_ONLY = 1
  };

// --- Inputs ---
input string TG_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN";
input string TG_CHAT = "YOUR_CHAT_ID";
input string InpModelSymbol = "";       // Empty = auto-detect from chart symbol
input bool InpUseTrainingThresholds = true;
input ENUM_SIGNAL_MODE InpSignalMode = SIGNAL_WITH_BOOSTER;
input double InpRiskPercent = 1.0;     // Risk 1% per trade
input int InpATRPeriod = 14;           // ATR Period
input double InpSL_ATR_Mult = 1.5;     // SL = ATR * Multiplier (Matched to AI Training)
input double InpTP_ATR_Mult = 2.0;     // TP = ATR * Multiplier (Matched to AI Training)
input double InpBaseConfidenceThreshold = 0.52; // Minimum base-model confidence (0.33 to 1.00)
input double InpBaseMinConfidenceEdge = 0.03;    // Minimum base-model gap between top-1 and top-2 probabilities
input double InpBoosterConfidenceThreshold = 0.55; // Minimum booster confidence (0.33 to 1.00)
input double InpBoosterMinConfidenceEdge = 0.02;   // Minimum booster gap between top-1 and top-2 probabilities
input double InpMaxSpreadPoints = 35.0;      // Maximum spread in points
input bool InpUseSessionFilter = false;      // Restrict trading to liquid hours
input int InpSessionStartHour = 6;           // Broker server hour start
input int InpSessionEndHour = 22;            // Broker server hour end
input bool InpUseMacdConfirmation = false;   // Optional MACD direction confirmation
input double InpMinAdx = 14.0;               // Regime strength threshold
input int InpCooldownBarsAfterLoss = 1;       // Pause after a losing trade
input bool InpUseBreakEven = false;          // Move SL to breakeven after favorable move
input double InpBreakEvenTriggerATR = 1.0;   // Trigger BE after 1 ATR in profit
input double InpBreakEvenLockPoints = 2.0;   // Lock a small profit after BE
input bool InpUseTrailing = false;     // Use Trailing Stop
input double InpTrail_ATR_Mult = 1.0;  // Trailing Stop in ATRs
input bool InpUseHardRegimeFilter = false; // If true, reject trades when regime is not aligned
input double InpRegimePenalty = 0.65;      // Position size multiplier when regime is weak
input ulong InpMagicNumber = 20260524;     // Magic number for this EA
input bool InpEnableTransactionCsv = true; // Log every deal to CSV
input string InpTransactionCsvFile = "TradingBot_Transactions.csv"; // Saved in Common\Files
input bool InpEnablePyramiding = false;    // Enable martingale/averaging basket logic
input double InpPyramidVolumeMultiplier = 2.0;
input int InpPyramidNum = 5;               // Additional entries after first position
input int InpPyramidSpreadRefreshSeconds = 313;

#resource "model_eurusd_H1.onnx" as uchar ExtModelEurUsd[]
#resource "model_usdjpy_H1.onnx" as uchar ExtModelUsdJpy[]
#resource "model_eurjpy_H1.onnx" as uchar ExtModelEurJpy[]
#resource "booster_eurusd_H1.onnx" as uchar BoosterModelEurUsd[]
#resource "booster_usdjpy_H1.onnx" as uchar BoosterModelUsdJpy[]
#resource "booster_eurjpy_H1.onnx" as uchar BoosterModelEurJpy[]
#resource "MQL5\\Files\\scaler_params.csv" as uchar ScalerParamsEurUsd[]
#resource "MQL5\\Files\\scaler_params_usdjpy_H1.csv" as uchar ScalerParamsUsdJpy[]
#resource "MQL5\\Files\\scaler_params_eurjpy_H1.csv" as uchar ScalerParamsEurJpy[]
#resource "training_report.json" as uchar TrainingReportEurUsd[]
#resource "training_report_usdjpy_H1.json" as uchar TrainingReportUsdJpy[]
#resource "training_report_eurjpy_H1.json" as uchar TrainingReportEurJpy[]
#resource "training_runs\\index_usdjpy_H1.jsonl" as uchar RunIndexUsdJpy[]
#resource "training_runs\\index_eurjpy_H1.jsonl" as uchar RunIndexEurJpy[]

#define SAMPLE_SIZE 24
#define NUM_FEATURES 25 // MACD Filter Strategy with Triple Barrier
#define BOOSTER_FEATURES 30
#define CLASS_UP 2
#define CLASS_SIDEWAYS 1
#define CLASS_DOWN 0

long ExtHandle = INVALID_HANDLE;
long BoosterHandle = INVALID_HANDLE;
int atrHandle, rsiHandle, macdHandle, bandsHandle, maHandle;
int stochHandle, adxHandle; // NEW
int rsiH4Handle, maH4Handle, rsiM15Handle, atrM15Handle;
CTrade trade;
double g_scalerCenter[NUM_FEATURES];
double g_scalerScale[NUM_FEATURES];
string g_modelSymbol = "";
string g_modelPairSlug = "";
string g_scalerResourceName = "";
bool g_boosterEnabled = false;
double g_baseConfidenceThreshold = 0.0;
double g_baseMinConfidenceEdge = 0.0;
double g_boosterConfidenceThreshold = 0.0;
double g_boosterMinConfidenceEdge = 0.0;
string g_baseThresholdSource = "input";
string g_boosterThresholdSource = "input";

int g_cooldownBarsLeft = 0;
ulong g_lastProcessedDeal = 0;
double g_pyramid_dis1 = 0.0;
double g_pyramid_dis2 = 0.0;
double g_pyramid_dis3 = 0.0;
double g_pyramid_dis4 = 0.0;
double g_pyramid_dis5 = 0.0;
double g_pyramid_max_spread = 0.0;
double g_pyramid_min_spread = DBL_MAX;
double g_pyramid_avg_spread = 0.0;
datetime g_pyramid_last_refresh = 0;

bool IsTradingSessionAllowed();
double GetCurrentSpreadPoints();
double NormalizeVolume(double lot);
double CalculateLot(double sl_points);
void UpdateCooldownFromHistory();
void ManageOpenPosition();
bool BuildBoosterInput(const float &baseProb[], const double &latestNormalizedFeatures[], float &boosterInput[]);
void PrintTradeFailure(const string side);
double SymbolPointSize(const string sym);
int CountManagedPositions(const string sym, const long type = -1);
bool HasManagedPositions(const string sym);
ulong GetFirstManagedPositionTicket(const string sym, const ENUM_POSITION_TYPE type);
ulong GetLastManagedPositionTicket(const string sym, const ENUM_POSITION_TYPE type);
double BasketProfit(const string sym, const ENUM_POSITION_TYPE type);
double BasketTargetProfit(const string sym, const ENUM_POSITION_TYPE type);
void CloseBasket(const string sym, const ENUM_POSITION_TYPE type);
void PyramidCloseBaskets(const string sym);
void UpdatePyramidDistances(const string sym, const datetime now);
bool PyramidAddPosition(const string sym, const ENUM_POSITION_TYPE type);
void PyramidManagePositions(const string sym);
bool ModifyPositionStopsByTicket(const ulong ticket, const double sl, const double tp, const string tag);
string DealTypeToText(const ENUM_DEAL_TYPE deal_type);
string DealEntryToText(const ENUM_DEAL_ENTRY deal_entry);
bool AppendDealCsv(const ulong deal_ticket);
bool ExportAllDealHistoryCsv();
string Trimmed(const string value);
string UpperTrimmed(const string value);
bool ResolvePairArtifacts(const string requestedSymbol, string &resolvedSymbol, string &pairSlug, string &scalerResourceName);
bool LoadScalerParamsFromResource(const uchar &resourceBuffer[]);
string ResourceToText(const uchar &resourceBuffer[]);
string LastNonEmptyLine(const string text);
bool TryExtractJsonNumber(const string text, const string anchor, const string key, double &value);
void LoadThresholdConfig();

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   string chartModelSymbol = "";
   string chartPairSlug = "";
   string chartScalerFile = "";
   if(!ResolvePairArtifacts(_Symbol, chartModelSymbol, chartPairSlug, chartScalerFile))
     {
      Print("Unsupported chart symbol for model selection: ", _Symbol);
      return(INIT_FAILED);
     }

   if(!ResolvePairArtifacts(InpModelSymbol == "" ? _Symbol : InpModelSymbol, g_modelSymbol, g_modelPairSlug, g_scalerResourceName))
     {
      Print("Unsupported InpModelSymbol/chart symbol combination: ", (InpModelSymbol == "" ? _Symbol : InpModelSymbol));
      return(INIT_FAILED);
     }

   if(g_modelSymbol != chartModelSymbol)
     {
      Print("Configured model symbol ", g_modelSymbol, " does not match chart/tester symbol family ", chartModelSymbol, ".");
      return(INIT_FAILED);
     }

   if(g_modelSymbol == "EURUSD")
     {
      ExtHandle = OnnxCreateFromBuffer(ExtModelEurUsd, ONNX_DEFAULT);
      BoosterHandle = OnnxCreateFromBuffer(BoosterModelEurUsd, ONNX_DEFAULT);
     }
   else if(g_modelSymbol == "USDJPY")
     {
      ExtHandle = OnnxCreateFromBuffer(ExtModelUsdJpy, ONNX_DEFAULT);
      BoosterHandle = OnnxCreateFromBuffer(BoosterModelUsdJpy, ONNX_DEFAULT);
     }
   else if(g_modelSymbol == "EURJPY")
     {
      ExtHandle = OnnxCreateFromBuffer(ExtModelEurJpy, ONNX_DEFAULT);
      BoosterHandle = OnnxCreateFromBuffer(BoosterModelEurJpy, ONNX_DEFAULT);
     }
   else
     {
      Print("No embedded model resource configured for ", g_modelSymbol);
      return(INIT_FAILED);
     }

   if(ExtHandle == INVALID_HANDLE)
     {
      Print("ONNX model creation failed for ", g_modelSymbol, ". Error: ", GetLastError());
      return(INIT_FAILED);
     }
     
   const long input_shape[] = {1, SAMPLE_SIZE, NUM_FEATURES};
   if(!OnnxSetInputShape(ExtHandle, ONNX_DEFAULT, input_shape))
     {
      Print("OnnxSetInputShape error ", GetLastError());
      return(INIT_FAILED);
     }
     
   const long output_shape[] = {1, 3};
   if(!OnnxSetOutputShape(ExtHandle, 0, output_shape))
     {
      Print("OnnxSetOutputShape error ", GetLastError());
      return(INIT_FAILED);
     }

   bool scalerLoaded = false;
   if(g_modelSymbol == "EURUSD")
      scalerLoaded = LoadScalerParamsFromResource(ScalerParamsEurUsd);
   else if(g_modelSymbol == "USDJPY")
      scalerLoaded = LoadScalerParamsFromResource(ScalerParamsUsdJpy);
   else if(g_modelSymbol == "EURJPY")
      scalerLoaded = LoadScalerParamsFromResource(ScalerParamsEurJpy);

   if(!scalerLoaded)
     {
      Print("Scaler parameter load failed for resource ", g_scalerResourceName);
      return(INIT_FAILED);
     }

   LoadThresholdConfig();

   if(BoosterHandle == INVALID_HANDLE)
     {
      g_boosterEnabled = false;
      Print("Booster ONNX model creation failed for ", g_modelSymbol, ". Error: ", GetLastError(), ". Proceeding with base model only.");
     }
   else
     {
      const long booster_input_shape[] = {1, BOOSTER_FEATURES};
      if(!OnnxSetInputShape(BoosterHandle, ONNX_DEFAULT, booster_input_shape))
        {
         Print("Booster OnnxSetInputShape error ", GetLastError(), ". Proceeding with base model only.");
         OnnxRelease(BoosterHandle);
         BoosterHandle = INVALID_HANDLE;
        }

      if(BoosterHandle != INVALID_HANDLE)
        {
         const long booster_output_shape[] = {1, 2};
         if(!OnnxSetOutputShape(BoosterHandle, 0, booster_output_shape))
            Print("Booster OnnxSetOutputShape warning ", GetLastError(), " - will fall back to base model if needed.");
        }
     }
   g_boosterEnabled = (BoosterHandle != INVALID_HANDLE);
   if(InpSignalMode == SIGNAL_BASE_ONLY)
      g_boosterEnabled = false;

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(20);
   trade.SetTypeFillingBySymbol(_Symbol);

   atrHandle = iATR(_Symbol, _Period, InpATRPeriod);
   rsiHandle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   macdHandle = iMACD(_Symbol, _Period, 12, 26, 9, PRICE_CLOSE);
   bandsHandle = iBands(_Symbol, _Period, 20, 0, 2.0, PRICE_CLOSE);
   maHandle = iMA(_Symbol, _Period, 24, 0, MODE_SMA, PRICE_CLOSE);
   stochHandle = iStochastic(_Symbol, _Period, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   adxHandle = iADX(_Symbol, _Period, 14);
   
   rsiH4Handle = iRSI(_Symbol, PERIOD_H4, 14, PRICE_CLOSE);
   maH4Handle = iMA(_Symbol, PERIOD_H4, 24, 0, MODE_SMA, PRICE_CLOSE);
   rsiM15Handle = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   atrM15Handle = iATR(_Symbol, PERIOD_M15, InpATRPeriod);
   
   if(atrHandle == INVALID_HANDLE || rsiHandle == INVALID_HANDLE || macdHandle == INVALID_HANDLE || bandsHandle == INVALID_HANDLE || maHandle == INVALID_HANDLE ||
      stochHandle == INVALID_HANDLE || adxHandle == INVALID_HANDLE ||
      rsiH4Handle == INVALID_HANDLE || maH4Handle == INVALID_HANDLE || rsiM15Handle == INVALID_HANDLE || atrM15Handle == INVALID_HANDLE)
     {
      Print("Failed to load indicators.");
     return(INIT_FAILED);
     }
   
   Print("Initialization successful. Model=", g_modelSymbol,
         " PairSlug=", g_modelPairSlug,
         " Booster=", (g_boosterEnabled ? "ON" : "OFF"),
         " ScalerResource=", g_scalerResourceName,
         " BaseThreshold=", DoubleToString(g_baseConfidenceThreshold, 2), "/", DoubleToString(g_baseMinConfidenceEdge, 2),
         " BaseSource=", g_baseThresholdSource,
         " BoosterThreshold=", DoubleToString(g_boosterConfidenceThreshold, 2), "/", DoubleToString(g_boosterMinConfidenceEdge, 2),
         " BoosterSource=", g_boosterThresholdSource,
         " SignalMode=", (InpSignalMode == SIGNAL_BASE_ONLY ? "BASE_ONLY" : "WITH_BOOSTER"));
   SendTelegram("✅ <b>Trading Bot Initialized</b>\nSymbol: " + _Symbol + "\nModel: " + g_modelSymbol);
   Print("CSV logging=", (InpEnableTransactionCsv ? "ON" : "OFF"),
         " file=", InpTransactionCsvFile,
         " location=Common\\Files",
         " pyramiding=", (InpEnablePyramiding ? "ON" : "OFF"),
         " pyramid_multiplier=", DoubleToString(InpPyramidVolumeMultiplier, 2),
         " pyramid_levels=", InpPyramidNum);
   
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
  void OnDeinit(const int reason)
  {
   if(InpEnableTransactionCsv)
      ExportAllDealHistoryCsv();

   if(ExtHandle != INVALID_HANDLE) OnnxRelease(ExtHandle);
   if(BoosterHandle != INVALID_HANDLE) OnnxRelease(BoosterHandle);
   IndicatorRelease(atrHandle);
   IndicatorRelease(rsiHandle);
   IndicatorRelease(macdHandle);
   IndicatorRelease(bandsHandle);
   IndicatorRelease(maHandle);
   IndicatorRelease(stochHandle);
   IndicatorRelease(adxHandle);
   IndicatorRelease(rsiH4Handle);
   IndicatorRelease(maH4Handle);
   IndicatorRelease(rsiM15Handle);
   IndicatorRelease(atrM15Handle);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, _Period, 0);
   
   if(currentBar != lastBar)
     {
      lastBar = currentBar;
      if(g_cooldownBarsLeft > 0)
        g_cooldownBarsLeft--;

      UpdateCooldownFromHistory();
      ExecuteTradingLogic();
     }
     
   if(InpEnablePyramiding)
     {
      UpdatePyramidDistances(_Symbol, TimeCurrent());
      PyramidCloseBaskets(_Symbol);
      PyramidManagePositions(_Symbol);
     }

   if(InpUseTrailing) ManageOpenPosition();
  }

//+------------------------------------------------------------------+
//| Core Trading Logic                                               |
//+------------------------------------------------------------------+
void ExecuteTradingLogic()
  {
   float input_data[];
   ArrayResize(input_data, SAMPLE_SIZE * NUM_FEATURES);
   
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   double rsi[], atr[], macdMain[], macdSignal[], bandsUpper[], bandsLower[], ma[];
   double stochMain[], stochSignal[], adxMain[], adxPlus[], adxMinus[];
   
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(macdMain, true);
   ArraySetAsSeries(macdSignal, true);
   ArraySetAsSeries(bandsUpper, true);
   ArraySetAsSeries(bandsLower, true);
   ArraySetAsSeries(ma, true);
   ArraySetAsSeries(stochMain, true);
   ArraySetAsSeries(stochSignal, true);
   ArraySetAsSeries(adxMain, true);
   ArraySetAsSeries(adxPlus, true);
   ArraySetAsSeries(adxMinus, true);

   if(InpUseSessionFilter && !IsTradingSessionAllowed())
     {
      Print("Trading blocked: outside configured session window.");
      return;
     }

   double spreadPoints = GetCurrentSpreadPoints();
   if(spreadPoints > InpMaxSpreadPoints)
     {
      Print("Trading blocked: spread too high (", DoubleToString(spreadPoints, 1), " points).");
      return;
     }

   if(g_cooldownBarsLeft > 0)
     {
      Print("Trading blocked: cooldown active for ", g_cooldownBarsLeft, " more bar(s).");
      return;
     }

   if(InpEnablePyramiding && HasManagedPositions(_Symbol))
     {
      Print("Trading blocked: pyramiding basket is active.");
      return;
     }

   if(CopyRates(_Symbol, _Period, 1, SAMPLE_SIZE + 1, rates) < SAMPLE_SIZE + 1) return;
   if(CopyBuffer(rsiHandle, 0, 1, SAMPLE_SIZE, rsi) < SAMPLE_SIZE) return;
   if(CopyBuffer(atrHandle, 0, 1, SAMPLE_SIZE, atr) < SAMPLE_SIZE) return;
   if(CopyBuffer(macdHandle, 0, 1, SAMPLE_SIZE, macdMain) < SAMPLE_SIZE) return;
   if(CopyBuffer(macdHandle, 1, 1, SAMPLE_SIZE, macdSignal) < SAMPLE_SIZE) return;
   if(CopyBuffer(bandsHandle, 1, 1, SAMPLE_SIZE, bandsUpper) < SAMPLE_SIZE) return;
   if(CopyBuffer(bandsHandle, 2, 1, SAMPLE_SIZE, bandsLower) < SAMPLE_SIZE) return;
   if(CopyBuffer(maHandle, 0, 1, SAMPLE_SIZE, ma) < SAMPLE_SIZE) return;
   if(CopyBuffer(stochHandle, 0, 1, SAMPLE_SIZE, stochMain) < SAMPLE_SIZE) return;
   if(CopyBuffer(stochHandle, 1, 1, SAMPLE_SIZE, stochSignal) < SAMPLE_SIZE) return;
   if(CopyBuffer(adxHandle, 0, 1, SAMPLE_SIZE, adxMain) < SAMPLE_SIZE) return;
   if(CopyBuffer(adxHandle, 1, 1, SAMPLE_SIZE, adxPlus) < SAMPLE_SIZE) return;
   if(CopyBuffer(adxHandle, 2, 1, SAMPLE_SIZE, adxMinus) < SAMPLE_SIZE) return;

   int ptr = 0;
   double latestRsiH4 = 0.0;
   double latestMaH4Rel = 0.0;
   double latestAdx = 0.0;
   double latestNormalizedFeatures[NUM_FEATURES];
   for(int i = SAMPLE_SIZE - 1; i >= 0; i--)
     {
      double prev_open = rates[i+1].open;
      double prev_high = rates[i+1].high;
      double prev_low = rates[i+1].low;
      double prev_close = rates[i+1].close;
      double curr_close = rates[i].close;

      int h4Shift = iBarShift(_Symbol, PERIOD_H4, rates[i].time, false);
      int m15Shift = iBarShift(_Symbol, PERIOD_M15, rates[i].time, false);
      if(h4Shift < 0 || m15Shift < 0)
        {
         Print("Unable to resolve MTF shift for ", TimeToString(rates[i].time));
         return;
        }

      double rsiH4_val[1], maH4_val[1], rsiM15_val[1], atrM15_val[1];
      if(CopyBuffer(rsiH4Handle, 0, h4Shift, 1, rsiH4_val) < 1 ||
         CopyBuffer(maH4Handle, 0, h4Shift, 1, maH4_val) < 1 ||
         CopyBuffer(rsiM15Handle, 0, m15Shift, 1, rsiM15_val) < 1 ||
         CopyBuffer(atrM15Handle, 0, m15Shift, 1, atrM15_val) < 1)
        {
         Print("Failed to copy MTF indicator values for ", TimeToString(rates[i].time));
         return;
        }
      
      MqlDateTime dt;
      TimeToStruct(rates[i].time, dt);

      double raw_features[NUM_FEATURES];
      raw_features[0] = (rates[i].open - prev_open) / prev_open;
      raw_features[1] = (rates[i].high - prev_high) / prev_high;
      raw_features[2] = (rates[i].low - prev_low) / prev_low;
      raw_features[3] = (curr_close - prev_close) / prev_close;
      raw_features[4] = MathLog((double)rates[i].tick_volume + 1.0);
      raw_features[5] = rsi[i];
      raw_features[6] = atr[i];
      raw_features[7] = macdMain[i];
      raw_features[8] = macdSignal[i];
      raw_features[9] = (bandsUpper[i] - curr_close) / curr_close;
      raw_features[10] = (bandsLower[i] - curr_close) / curr_close;
      raw_features[11] = stochMain[i];
      raw_features[12] = stochSignal[i];
      raw_features[13] = adxMain[i];
      raw_features[14] = adxPlus[i];
      raw_features[15] = adxMinus[i];
      raw_features[16] = rsiH4_val[0];
      raw_features[17] = (curr_close - maH4_val[0]) / curr_close;
      raw_features[18] = rsiM15_val[0];
      raw_features[19] = atrM15_val[0];
      raw_features[20] = MathSin(2 * M_PI * dt.hour / 24.0);
      raw_features[21] = MathCos(2 * M_PI * dt.hour / 24.0);
      raw_features[22] = MathSin(2 * M_PI * dt.day_of_week / 7.0);
      raw_features[23] = MathCos(2 * M_PI * dt.day_of_week / 7.0);
      raw_features[24] = (curr_close - ma[i]) / curr_close;

      if(i == 0)
        {
         latestRsiH4 = rsiH4_val[0];
         latestMaH4Rel = raw_features[17];
         latestAdx = adxMain[i];
        }
      
      for(int f=0; f<NUM_FEATURES; f++)
        {
         double normalized = (raw_features[f] - g_scalerCenter[f]) / g_scalerScale[f];
         input_data[ptr] = (float)normalized;
         if(i == 0)
            latestNormalizedFeatures[f] = normalized;
         ptr++;
        }
     }
   
   float output_data[3]; 
   
   if(!OnnxRun(ExtHandle, ONNX_NO_CONVERSION, input_data, output_data))
     {
      Print("ONNX Run failed. Error: ", GetLastError());
      return;
   }
   
   int baseClass = CLASS_DOWN;
   float maxProb = output_data[0];
   float secondProb = output_data[1];
   int maxIndex = 0;

   if(output_data[1] > maxProb)
     {
      secondProb = maxProb;
      maxProb = output_data[1];
      maxIndex = 1;
     }
   else
     {
      secondProb = output_data[1];
     }

   if(output_data[2] > maxProb)
     {
      secondProb = maxProb;
      maxProb = output_data[2];
      maxIndex = 2;
     }
   else if(output_data[2] > secondProb)
     {
      secondProb = output_data[2];
     }

   baseClass = maxIndex;
   float baseEdge = maxProb - secondProb;

   Print("Base AI: DOWN=", output_data[0], " SIDEWAYS=", output_data[1], " UP=", output_data[2], " -> Class: ", baseClass, " Conf=", maxProb, " Edge=", baseEdge);

   if(maxProb < g_baseConfidenceThreshold || baseEdge < g_baseMinConfidenceEdge) 
     {
      Print("BASE REJECTS: Confidence too low or edge too small.");
      return;
     }

   float boosterInput[BOOSTER_FEATURES];
   boosterInput[0] = output_data[0];
   boosterInput[1] = output_data[1];
   boosterInput[2] = output_data[2];
   boosterInput[3] = maxProb;
   boosterInput[4] = baseEdge;
   for(int f = 0; f < NUM_FEATURES; f++)
      boosterInput[5 + f] = (float)latestNormalizedFeatures[f];

   float boosterSkipProb = 0.5f;
   float boosterEnterProb = 0.5f;
   bool boosterAvailable = false;
   float boosterOutput[1][2];
   if(g_boosterEnabled && BoosterHandle != INVALID_HANDLE && OnnxRun(BoosterHandle, ONNX_NO_CONVERSION, boosterInput, boosterOutput))
     {
      boosterAvailable = true;
      boosterSkipProb = boosterOutput[0][0];
      boosterEnterProb = boosterOutput[0][1];
     }
   else if(g_boosterEnabled)
     {
      Print("Booster ONNX Run failed. Error: ", GetLastError(), ". Proceeding with base model only.");
     }

   float boosterEdge = boosterEnterProb - boosterSkipProb;
   Print("Booster AI: SKIP=", boosterSkipProb, " ENTER=", boosterEnterProb, " -> Gate=", (boosterEnterProb >= boosterSkipProb ? "ENTER" : "SKIP"), " Conf=", boosterEnterProb, " Edge=", boosterEdge);

   bool boosterAllows = true;
   double boosterPenalty = 1.0;
   if(boosterAvailable)
     {
      boosterAllows = (boosterEnterProb >= boosterSkipProb) && (boosterEnterProb >= g_boosterConfidenceThreshold) && (boosterEdge >= g_boosterMinConfidenceEdge);
      boosterPenalty = boosterAllows ? 1.0 : 0.45;
      if(!boosterAllows)
         Print("BOOSTER LOW CONFIDENCE: proceeding with reduced size.");
     }
     
   double currentAtr[];
   if(CopyBuffer(atrHandle, 0, 0, 1, currentAtr) < 1) return;
   
   double sl_dist = currentAtr[0] * InpSL_ATR_Mult;
   double tp_dist = currentAtr[0] * InpTP_ATR_Mult;
   double minStopDistance = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(minStopDistance > 0.0)
     {
      sl_dist = MathMax(sl_dist, minStopDistance);
      tp_dist = MathMax(tp_dist, minStopDistance);
     }
   
   bool regimeBuy = (latestAdx > InpMinAdx && latestRsiH4 > 50.0 && latestMaH4Rel > 0.0);
   bool regimeSell = (latestAdx > InpMinAdx && latestRsiH4 < 50.0 && latestMaH4Rel < 0.0);
   bool regimeAlignedBuy = regimeBuy;
   bool regimeAlignedSell = regimeSell;
   double macdDirection = macdMain[0] - macdSignal[0];
   bool macdBull = macdDirection > 0.0;
   bool macdBear = macdDirection < 0.0;

   if(baseClass == CLASS_UP)
     {
      if(!HasManagedPositions(_Symbol))
        {
         if(InpUseMacdConfirmation && !macdBull)
           {
            Print("BUY rejected: MACD confirmation failed.");
            return;
           }

         if(InpUseHardRegimeFilter && !regimeAlignedBuy)
           {
            Print("BUY rejected: hard regime filter failed.");
            return;
           }

         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = ask - sl_dist;
         double tp = ask + tp_dist;
         double lot = CalculateLot(sl_dist);
         lot *= boosterPenalty;
         if(!regimeAlignedBuy)
            lot *= InpRegimePenalty;
         if(lot > 0.0)
           {
            if(trade.Buy(lot, _Symbol, ask, sl, tp, "AI Regime Buy"))
              SendTelegram("✅ <b>BUY CONFIRMED</b>\nSymbol: " + _Symbol + "\nLot: " + DoubleToString(lot, 2) + "\nConfidence: " + DoubleToString(maxProb*100, 1) + "%");
            else
               PrintTradeFailure("BUY");
           }
        }
      else
        {
         Print("BUY rejected: existing managed position is active.");
        }
     }
   else if(baseClass == CLASS_DOWN)
     {
      if(!HasManagedPositions(_Symbol))
        {
         if(InpUseMacdConfirmation && !macdBear)
           {
            Print("SELL rejected: MACD confirmation failed.");
            return;
           }

         if(InpUseHardRegimeFilter && !regimeAlignedSell)
           {
            Print("SELL rejected: hard regime filter failed.");
            return;
           }

         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = bid + sl_dist;
         double tp = bid - tp_dist;
         double lot = CalculateLot(sl_dist);
         lot *= boosterPenalty;
         if(!regimeAlignedSell)
            lot *= InpRegimePenalty;
         if(lot > 0.0)
           {
            if(trade.Sell(lot, _Symbol, bid, sl, tp, "AI Regime Sell"))
              SendTelegram("✅ <b>SELL CONFIRMED</b>\nSymbol: " + _Symbol + "\nLot: " + DoubleToString(lot, 2) + "\nConfidence: " + DoubleToString(maxProb*100, 1) + "%");
            else
               PrintTradeFailure("SELL");
           }
        }
      else
        {
         Print("SELL rejected: existing managed position is active.");
        }
     }
  }

//+------------------------------------------------------------------+
//| Trade Failure Diagnostics                                        |
//+------------------------------------------------------------------+
void PrintTradeFailure(const string side)
  {
   Print(side, " order failed. Retcode=", trade.ResultRetcode(),
         " Desc=", trade.ResultRetcodeDescription(),
         " Comment=", trade.ResultComment(),
         " LastError=", GetLastError());
  }

bool BuildBoosterInput(const float &baseProb[], const double &latestNormalizedFeatures[], float &boosterInput[])
  {
   return false;
  }

string Trimmed(const string value)
  {
   string text = value;
   StringTrimLeft(text);
   StringTrimRight(text);
   return(text);
  }

string UpperTrimmed(const string value)
  {
   string text = Trimmed(value);
   StringToUpper(text);
   return(text);
  }

bool ResolvePairArtifacts(const string requestedSymbol, string &resolvedSymbol, string &pairSlug, string &scalerResourceName)
  {
   string symbolKey = UpperTrimmed(requestedSymbol);
   if(symbolKey == "")
      symbolKey = UpperTrimmed(_Symbol);

   if(StringFind(symbolKey, "EURUSD") >= 0)
     {
      resolvedSymbol = "EURUSD";
      pairSlug = "eurusd_H1";
      scalerResourceName = "ScalerParamsEurUsd";
      return(true);
     }

   if(StringFind(symbolKey, "USDJPY") >= 0)
     {
      resolvedSymbol = "USDJPY";
      pairSlug = "usdjpy_H1";
      scalerResourceName = "ScalerParamsUsdJpy";
      return(true);
     }

   if(StringFind(symbolKey, "EURJPY") >= 0)
     {
      resolvedSymbol = "EURJPY";
      pairSlug = "eurjpy_H1";
      scalerResourceName = "ScalerParamsEurJpy";
      return(true);
     }

   return(false);
  }

bool LoadScalerParamsFromResource(const uchar &resourceBuffer[])
  {
   string csvText = CharArrayToString(resourceBuffer);
   if(csvText == "")
      return(false);

   string normalized = csvText;
   StringReplace(normalized, "\r\n", "\n");
   StringReplace(normalized, "\r", "\n");

   string rows[];
   int rowCount = StringSplit(normalized, '\n', rows);
   if(rowCount < 2)
      return(false);

   string centerCols[];
   string scaleCols[];
   int centerCount = StringSplit(rows[0], ',', centerCols);
   int scaleCount = StringSplit(rows[1], ',', scaleCols);
   if(centerCount < NUM_FEATURES || scaleCount < NUM_FEATURES)
      return(false);

   for(int i = 0; i < NUM_FEATURES; i++)
     {
      g_scalerCenter[i] = StringToDouble(centerCols[i]);
    }

   for(int i = 0; i < NUM_FEATURES; i++)
     {
      g_scalerScale[i] = StringToDouble(scaleCols[i]);
      if(g_scalerScale[i] == 0.0)
         g_scalerScale[i] = 1.0;
     }

   return(true);
  }

string ResourceToText(const uchar &resourceBuffer[])
  {
   return(CharArrayToString(resourceBuffer));
  }

string LastNonEmptyLine(const string text)
  {
   string normalized = text;
   StringReplace(normalized, "\r\n", "\n");
   StringReplace(normalized, "\r", "\n");

   string rows[];
   int rowCount = StringSplit(normalized, '\n', rows);
   for(int i = rowCount - 1; i >= 0; i--)
     {
      string line = Trimmed(rows[i]);
      if(line != "")
         return(line);
     }
   return("");
  }

bool TryExtractJsonNumber(const string text, const string anchor, const string key, double &value)
  {
   int anchorPos = StringFind(text, anchor);
   if(anchorPos < 0)
      return(false);

   int keyPos = StringFind(text, key, anchorPos);
   if(keyPos < 0)
      return(false);

   int colonPos = StringFind(text, ":", keyPos);
   if(colonPos < 0)
      return(false);

   string suffix = StringSubstr(text, colonPos + 1);
   suffix = Trimmed(suffix);

   int stopPos = StringLen(suffix);
   int commaPos = StringFind(suffix, ",");
   int bracePos = StringFind(suffix, "}");
   if(commaPos >= 0 && commaPos < stopPos)
      stopPos = commaPos;
   if(bracePos >= 0 && bracePos < stopPos)
      stopPos = bracePos;

   string numberText = Trimmed(StringSubstr(suffix, 0, stopPos));
   if(numberText == "" || numberText == "null")
      return(false);

   value = StringToDouble(numberText);
   return(true);
  }

void LoadThresholdConfig()
  {
   g_baseConfidenceThreshold = InpBaseConfidenceThreshold;
   g_baseMinConfidenceEdge = InpBaseMinConfidenceEdge;
   g_boosterConfidenceThreshold = InpBoosterConfidenceThreshold;
   g_boosterMinConfidenceEdge = InpBoosterMinConfidenceEdge;
   g_baseThresholdSource = "input";
   g_boosterThresholdSource = "input";

   if(!InpUseTrainingThresholds)
      return;

   string runIndexText = "";
   string reportText = "";
   if(g_modelSymbol == "USDJPY")
     {
      runIndexText = LastNonEmptyLine(ResourceToText(RunIndexUsdJpy));
      reportText = ResourceToText(TrainingReportUsdJpy);
     }
   else if(g_modelSymbol == "EURJPY")
     {
      runIndexText = LastNonEmptyLine(ResourceToText(RunIndexEurJpy));
      reportText = ResourceToText(TrainingReportEurJpy);
     }
   else if(g_modelSymbol == "EURUSD")
     {
      reportText = ResourceToText(TrainingReportEurUsd);
     }

   double parsedValue = 0.0;
   if(runIndexText != "")
     {
      bool hasBaseConf = TryExtractJsonNumber(runIndexText, "\"best_validation_metrics\"", "\"tuned_confidence_threshold\"", parsedValue);
      if(hasBaseConf)
        {
         g_baseConfidenceThreshold = parsedValue;
         g_baseThresholdSource = "training_index";
        }

      bool hasBaseEdge = TryExtractJsonNumber(runIndexText, "\"best_validation_metrics\"", "\"tuned_min_confidence_edge\"", parsedValue);
      if(hasBaseEdge)
        {
         g_baseMinConfidenceEdge = parsedValue;
         g_baseThresholdSource = "training_index";
        }
     }

   if(reportText != "")
     {
      bool hasBaseConf = TryExtractJsonNumber(reportText, "\"threshold_config\":", "\"confidence_threshold\"", parsedValue);
      if(hasBaseConf)
        {
         g_baseConfidenceThreshold = parsedValue;
         g_baseThresholdSource = "training_report";
        }

      bool hasBaseEdge = TryExtractJsonNumber(reportText, "\"threshold_config\":", "\"min_confidence_edge\"", parsedValue);
      if(hasBaseEdge)
        {
         g_baseMinConfidenceEdge = parsedValue;
         g_baseThresholdSource = "training_report";
        }

      bool hasBoosterConf = TryExtractJsonNumber(reportText, "\"booster\":", "\"tuned_confidence_threshold\"", parsedValue);
      if(hasBoosterConf)
        {
         g_boosterConfidenceThreshold = parsedValue;
         g_boosterThresholdSource = "training_report";
        }

      bool hasBoosterEdge = TryExtractJsonNumber(reportText, "\"booster\":", "\"tuned_min_confidence_edge\"", parsedValue);
      if(hasBoosterEdge)
        {
         g_boosterMinConfidenceEdge = parsedValue;
         g_boosterThresholdSource = "training_report";
        }
     }

   if(g_baseThresholdSource == "input" && g_modelSymbol == "EURJPY" && runIndexText != "")
     {
      bool hasBaseConfFromIndex = TryExtractJsonNumber(runIndexText, "\"best_validation_metrics\"", "\"tuned_confidence_threshold\"", parsedValue);
      if(hasBaseConfFromIndex)
        {
         g_baseConfidenceThreshold = parsedValue;
         g_baseThresholdSource = "training_index";
        }

      bool hasBaseEdgeFromIndex = TryExtractJsonNumber(runIndexText, "\"best_validation_metrics\"", "\"tuned_min_confidence_edge\"", parsedValue);
      if(hasBaseEdgeFromIndex)
        {
         g_baseMinConfidenceEdge = parsedValue;
         g_baseThresholdSource = "training_index";
        }
     }
  }

//+------------------------------------------------------------------+
//| Pyramiding / Basket Helpers                                      |
//+------------------------------------------------------------------+
double SymbolPointSize(const string sym)
  {
   double point = 0.0;
   if(!SymbolInfoDouble(sym, SYMBOL_POINT, point) || point <= 0.0)
      return _Point;
   return point;
  }

int CountManagedPositions(const string sym, const long type = -1)
  {
   int total = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != sym)
         continue;
      if(type != -1 && (long)PositionGetInteger(POSITION_TYPE) != type)
         continue;
      total++;
     }
   return total;
  }

bool HasManagedPositions(const string sym)
  {
   return (CountManagedPositions(sym) > 0);
  }

ulong GetFirstManagedPositionTicket(const string sym, const ENUM_POSITION_TYPE type)
  {
   ulong first_ticket = 0;
   datetime first_time = LONG_MAX;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != sym)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != type)
         continue;
      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      if(open_time < first_time)
        {
         first_time = open_time;
         first_ticket = ticket;
        }
     }
   return first_ticket;
  }

ulong GetLastManagedPositionTicket(const string sym, const ENUM_POSITION_TYPE type)
  {
   ulong last_ticket = 0;
   datetime last_time = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != sym)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != type)
         continue;
      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      if(open_time > last_time)
        {
         last_time = open_time;
         last_ticket = ticket;
        }
     }
   return last_ticket;
  }

double BasketProfit(const string sym, const ENUM_POSITION_TYPE type)
  {
   double profit = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != sym)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != type)
         continue;
      profit += PositionGetDouble(POSITION_PROFIT);
     }
   return profit;
  }

double BasketTargetProfit(const string sym, const ENUM_POSITION_TYPE type)
  {
   ulong first_ticket = GetFirstManagedPositionTicket(sym, type);
   if(first_ticket == 0 || !PositionSelectByTicket(first_ticket))
      return 0.0;
   int total = CountManagedPositions(sym, type);
   if(total <= 0)
      return 0.0;
   double first_lot = PositionGetDouble(POSITION_VOLUME);
   if(first_lot <= 0.0)
      return 0.0;
   return NormalizeDouble((first_lot * g_pyramid_dis2) / total, 2);
  }

void CloseBasket(const string sym, const ENUM_POSITION_TYPE type)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != sym)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != type)
         continue;
      if(!trade.PositionClose(ticket))
         PrintTradeFailure("BASKET CLOSE");
     }
  }

void PyramidCloseBaskets(const string sym)
  {
   if(!InpEnablePyramiding)
      return;
   ENUM_POSITION_TYPE types[2] = {POSITION_TYPE_BUY, POSITION_TYPE_SELL};
   for(int i = 0; i < 2; i++)
     {
      ENUM_POSITION_TYPE type = types[i];
      int count = CountManagedPositions(sym, type);
      if(count <= 0)
         continue;
      double target = BasketTargetProfit(sym, type);
      double profit = BasketProfit(sym, type);
      if(target > 0.0 && profit >= target)
        {
         Print("[BASKET CLOSE] sym=", sym,
               " type=", (type == POSITION_TYPE_BUY ? "BUY" : "SELL"),
               " count=", count,
               " profit=", DoubleToString(profit, 2),
               " target=", DoubleToString(target, 2));
         CloseBasket(sym, type);
        }
     }
  }

void UpdatePyramidDistances(const string sym, const datetime now)
  {
   if(!InpEnablePyramiding)
      return;

   double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
   double bid = SymbolInfoDouble(sym, SYMBOL_BID);
   double point = SymbolPointSize(sym);
   if(ask <= 0.0 || bid <= 0.0 || point <= 0.0)
      return;

   double spread_points = (ask - bid) / point;
   if(spread_points <= 0.0)
      spread_points = 1.0;

   if(g_pyramid_last_refresh == 0 || (now - g_pyramid_last_refresh) >= InpPyramidSpreadRefreshSeconds)
     {
      g_pyramid_max_spread = spread_points;
      g_pyramid_min_spread = spread_points;
      g_pyramid_last_refresh = now;
     }

   if(spread_points > g_pyramid_max_spread)
      g_pyramid_max_spread = spread_points;
   if(spread_points < g_pyramid_min_spread)
      g_pyramid_min_spread = spread_points;

   g_pyramid_avg_spread = (g_pyramid_max_spread + g_pyramid_min_spread) / 2.0;
   double safe_min_spread = MathMax(1.0, g_pyramid_min_spread);
   g_pyramid_dis1 = (int)(MathSqrt(g_pyramid_avg_spread) * (MathSqrt(safe_min_spread) + 55.0));
   g_pyramid_dis2 = (int)(MathSqrt(g_pyramid_avg_spread) * (MathSqrt(safe_min_spread) + 89.0));
   g_pyramid_dis3 = (int)(MathSqrt(g_pyramid_avg_spread) * (MathSqrt(g_pyramid_avg_spread) + 144.0));
   g_pyramid_dis4 = (int)(MathSqrt(g_pyramid_max_spread) * (MathSqrt(g_pyramid_avg_spread) + 233.0));
   g_pyramid_dis5 = (int)(MathSqrt(g_pyramid_max_spread) * (MathSqrt(g_pyramid_max_spread) + 377.0));
  }

bool PyramidAddPosition(const string sym, const ENUM_POSITION_TYPE type)
  {
   if(!InpEnablePyramiding)
      return false;

   int count = CountManagedPositions(sym, type);
   if(count <= 0)
      return false;

   int max_additions = MathMin(MathMax(InpPyramidNum, 0), 5);
   int additions_done = count - 1;
   if(additions_done >= max_additions)
      return false;

   ulong first_ticket = GetFirstManagedPositionTicket(sym, type);
   ulong last_ticket = GetLastManagedPositionTicket(sym, type);
   if(first_ticket == 0 || last_ticket == 0)
      return false;
   if(!PositionSelectByTicket(first_ticket))
      return false;

   double first_open = PositionGetDouble(POSITION_PRICE_OPEN);
   if(!PositionSelectByTicket(last_ticket))
      return false;
   double last_volume = PositionGetDouble(POSITION_VOLUME);
   if(first_open <= 0.0 || last_volume <= 0.0)
      return false;

   double point = SymbolPointSize(sym);
   if(point <= 0.0)
      return false;

   double current_price = (type == POSITION_TYPE_BUY)
                          ? SymbolInfoDouble(sym, SYMBOL_BID)
                          : SymbolInfoDouble(sym, SYMBOL_ASK);
   if(current_price <= 0.0)
      return false;

   double threshold = (double)(g_pyramid_dis1);
   if(additions_done + 1 == 2) threshold = g_pyramid_dis2;
   else if(additions_done + 1 == 3) threshold = g_pyramid_dis3;
   else if(additions_done + 1 == 4) threshold = g_pyramid_dis4;
   else if(additions_done + 1 >= 5) threshold = g_pyramid_dis5;
   double threshold_price = threshold * point;

   double adverse_move = (type == POSITION_TYPE_BUY)
                         ? (first_open - current_price)
                         : (current_price - first_open);
   if(adverse_move < threshold_price)
      return false;

   double next_lot = NormalizeVolume(last_volume * InpPyramidVolumeMultiplier);
   if(next_lot <= 0.0)
      return false;

   string comment = StringFormat("PYR|%s|L%d", TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS), additions_done + 1);
   bool ok = false;
   if(type == POSITION_TYPE_BUY)
      ok = trade.Buy(next_lot, sym, 0.0, 0.0, 0.0, comment);
   else
      ok = trade.Sell(next_lot, sym, 0.0, 0.0, 0.0, comment);

   if(ok)
      Print("[PYRAMID OPEN] sym=", sym,
            " type=", (type == POSITION_TYPE_BUY ? "BUY" : "SELL"),
            " level=", additions_done + 1,
            " lot=", DoubleToString(next_lot, 2),
            " adverse_points=", DoubleToString(adverse_move / point, 1),
            " threshold_points=", DoubleToString(threshold, 1));
   else
      Print("[PYRAMID FAILED] sym=", sym,
            " type=", (type == POSITION_TYPE_BUY ? "BUY" : "SELL"),
            " level=", additions_done + 1,
            " lot=", DoubleToString(next_lot, 2),
            " retcode=", trade.ResultRetcode(),
            " desc=", trade.ResultRetcodeDescription());
   return ok;
  }

void PyramidManagePositions(const string sym)
  {
   if(!InpEnablePyramiding)
      return;
   if(CountManagedPositions(sym, POSITION_TYPE_BUY) > 0)
      PyramidAddPosition(sym, POSITION_TYPE_BUY);
   if(CountManagedPositions(sym, POSITION_TYPE_SELL) > 0)
      PyramidAddPosition(sym, POSITION_TYPE_SELL);
  }

bool ModifyPositionStopsByTicket(const ulong ticket, const double sl, const double tp, const string tag)
  {
   if(ticket == 0 || !PositionSelectByTicket(ticket))
      return false;

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.symbol = PositionGetString(POSITION_SYMBOL);
   request.magic = (ulong)PositionGetInteger(POSITION_MAGIC);
   request.sl = sl;
   request.tp = tp;

   if(!OrderSend(request, result))
     {
      Print(tag, " modify failed. LastError=", GetLastError(),
            " retcode=", result.retcode,
            " comment=", result.comment);
      return false;
     }

   if(result.retcode != TRADE_RETCODE_DONE && result.retcode != TRADE_RETCODE_PLACED)
     {
      Print(tag, " modify failed. retcode=", result.retcode,
            " comment=", result.comment);
      return false;
     }

   return true;
  }

//+------------------------------------------------------------------+
//| Transaction CSV Logging                                          |
//+------------------------------------------------------------------+
string DealTypeToText(const ENUM_DEAL_TYPE deal_type)
  {
   if(deal_type == DEAL_TYPE_BUY)
      return "BUY";
   if(deal_type == DEAL_TYPE_SELL)
      return "SELL";
   return "UNKNOWN";
  }

string DealEntryToText(const ENUM_DEAL_ENTRY deal_entry)
  {
   if(deal_entry == DEAL_ENTRY_IN)
      return "IN";
   if(deal_entry == DEAL_ENTRY_OUT)
      return "OUT";
   if(deal_entry == DEAL_ENTRY_INOUT)
      return "INOUT";
   if(deal_entry == DEAL_ENTRY_OUT_BY)
      return "OUT_BY";
   return "UNKNOWN";
  }

bool AppendDealCsv(const ulong deal_ticket)
  {
   if(!InpEnableTransactionCsv || deal_ticket == 0)
      return false;
   if(!HistoryDealSelect(deal_ticket))
      return false;

   string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
   if(symbol != _Symbol)
      return true;
   if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
      return true;

   int fh = FileOpen(InpTransactionCsvFile, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(fh == INVALID_HANDLE)
     {
      fh = FileOpen(InpTransactionCsvFile, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
      if(fh == INVALID_HANDLE)
        {
         Print("ERROR: cannot open transaction CSV: ", InpTransactionCsvFile, " err=", GetLastError());
         return false;
        }
     }

   if(FileSize(fh) == 0)
     {
      FileWrite(fh,
                "time",
                "deal_ticket",
                "order_ticket",
                "position_id",
                "symbol",
                "deal_type",
                "entry",
                "volume",
                "price",
                "sl",
                "tp",
                "profit",
                "swap",
                "commission",
                "comment");
     }

   FileSeek(fh, 0, SEEK_END);
   datetime deal_time = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
   string comment = HistoryDealGetString(deal_ticket, DEAL_COMMENT);
   StringReplace(comment, ",", ";");

   FileWrite(fh,
             TimeToString(deal_time, TIME_DATE | TIME_SECONDS),
             (string)deal_ticket,
             (string)HistoryDealGetInteger(deal_ticket, DEAL_ORDER),
             (string)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID),
             symbol,
             DealTypeToText((ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE)),
             DealEntryToText((ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_VOLUME), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PRICE), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SL), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_TP), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PROFIT), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SWAP), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION), 2),
             comment);

   FileClose(fh);
   return true;
  }

//+------------------------------------------------------------------+
//| Dynamic Position Sizing (Risk Management)                        |
//+------------------------------------------------------------------+
double CalculateLot(double sl_points)
  {
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * InpRiskPercent / 100.0;
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(sl_points <= 0 || tickValue <= 0 || tickSize <= 0) return NormalizeVolume(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));

   double riskPerLot = (sl_points / tickSize) * tickValue;
   if(riskPerLot <= 0.0) return NormalizeVolume(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));

   double lot = riskMoney / riskPerLot;
   return NormalizeVolume(lot);
  }

//+------------------------------------------------------------------+
//| Volume Normalization                                             |
//+------------------------------------------------------------------+
double NormalizeVolume(double lot)
  {
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(step <= 0.0)
      step = minLot;

   lot = MathMax(minLot, MathMin(maxLot, lot));
   lot = MathFloor(lot / step) * step;
   lot = MathMax(minLot, MathMin(maxLot, lot));
   return lot;
  }

//+------------------------------------------------------------------+
//| Session / Spread Filters                                         |
//+------------------------------------------------------------------+
bool IsTradingSessionAllowed()
  {
   if(!InpUseSessionFilter)
      return true;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   if(InpSessionStartHour == InpSessionEndHour)
      return true;

   if(InpSessionStartHour < InpSessionEndHour)
      return (dt.hour >= InpSessionStartHour && dt.hour < InpSessionEndHour);

   return (dt.hour >= InpSessionStartHour || dt.hour < InpSessionEndHour);
  }

double GetCurrentSpreadPoints()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0 || _Point <= 0.0)
      return 0.0;
   return (ask - bid) / _Point;
  }

void UpdateCooldownFromHistory()
  {
   datetime now = TimeCurrent();
   datetime from = now - 86400 * 30;
   if(!HistorySelect(from, now))
      return;

   int total = HistoryDealsTotal();
   ulong newestProcessed = g_lastProcessedDeal;

   for(int i = 0; i < total; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket <= g_lastProcessedDeal)
         continue;

      string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
      if(symbol != _Symbol)
         continue;

      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY)
         continue;

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) +
                      HistoryDealGetDouble(ticket, DEAL_SWAP) +
                      HistoryDealGetDouble(ticket, DEAL_COMMISSION);

      if(ticket > newestProcessed)
         newestProcessed = ticket;

      if(profit < 0.0)
        {
         g_cooldownBarsLeft = InpCooldownBarsAfterLoss;
         Print("Cooldown activated after losing trade. Bars left: ", g_cooldownBarsLeft);
        }
     }

   g_lastProcessedDeal = newestProcessed;
  }

void ManageOpenPosition()
  {
   double atr[];
   if(CopyBuffer(atrHandle, 0, 0, 1, atr) < 1)
      return;

   double trail = atr[0] * InpTrail_ATR_Mult;
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double minStopDistance = (double)MathMax(SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL),
                                            SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL)) * _Point;
   if(minStopDistance < _Point)
      minStopDistance = _Point;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;

      long type = PositionGetInteger(POSITION_TYPE);
      double price = SymbolInfoDouble(_Symbol, type == POSITION_TYPE_BUY ? SYMBOL_BID : SYMBOL_ASK);
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);

      if(type == POSITION_TYPE_BUY)
        {
         double desiredSL = sl;
         double trailSL = price - trail;
         if(trailSL > desiredSL)
            desiredSL = trailSL;

         if(InpUseBreakEven && (price - openPrice) >= (atr[0] * InpBreakEvenTriggerATR))
           {
            double breakEvenSL = openPrice + (InpBreakEvenLockPoints * _Point);
            if(breakEvenSL > desiredSL)
               desiredSL = breakEvenSL;
           }

         desiredSL = NormalizeDouble(desiredSL, digits);
         if(desiredSL > 0.0 && desiredSL > sl + (_Point * 0.5) && desiredSL < (price - minStopDistance))
           {
            if(!ModifyPositionStopsByTicket(ticket, desiredSL, tp, "TRAIL BUY"))
               PrintTradeFailure("TRAIL BUY");
           }
        }
      else if(type == POSITION_TYPE_SELL)
        {
         double desiredSL = sl;
         double trailSL = price + trail;
         if(desiredSL <= 0.0 || trailSL < desiredSL)
            desiredSL = trailSL;

         if(InpUseBreakEven && (openPrice - price) >= (atr[0] * InpBreakEvenTriggerATR))
           {
            double breakEvenSL = openPrice - (InpBreakEvenLockPoints * _Point);
            if(desiredSL <= 0.0 || breakEvenSL < desiredSL)
               desiredSL = breakEvenSL;
           }

         desiredSL = NormalizeDouble(desiredSL, digits);
         if(desiredSL > 0.0 && (sl <= 0.0 || desiredSL < sl - (_Point * 0.5)) && desiredSL > (price + minStopDistance))
           {
            if(!ModifyPositionStopsByTicket(ticket, desiredSL, tp, "TRAIL SELL"))
               PrintTradeFailure("TRAIL SELL");
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Telegram Notification                                            |
//+------------------------------------------------------------------+
void SendTelegram(string text)
  {
   string url = "https://api.telegram.org/bot" + TG_TOKEN + "/sendMessage?chat_id=" + TG_CHAT + "&text=" + text + "&parse_mode=HTML";
   char data[], result[];
   string headers;
   WebRequest("GET", url, NULL, NULL, 5000, data, 0, result, headers);
  }

bool ExportAllDealHistoryCsv()
  {
   if(!HistorySelect(0, TimeCurrent()))
      return false;

   int fh = FileOpen(InpTransactionCsvFile, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(fh == INVALID_HANDLE)
     {
         Print("ERROR: cannot rewrite transaction CSV: ", InpTransactionCsvFile, " err=", GetLastError());
      return false;
     }

   FileWrite(fh,
             "time",
             "deal_ticket",
             "order_ticket",
             "position_id",
             "symbol",
             "deal_type",
             "entry",
             "volume",
             "price",
             "sl",
             "tp",
             "profit",
             "swap",
             "commission",
             "comment");

   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0 || !HistoryDealSelect(deal_ticket))
         continue;

      string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      if(symbol != _Symbol)
         continue;
      if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
         continue;

      string comment = HistoryDealGetString(deal_ticket, DEAL_COMMENT);
      StringReplace(comment, ",", ";");
      datetime deal_time = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);

      FileWrite(fh,
                TimeToString(deal_time, TIME_DATE | TIME_SECONDS),
                (string)deal_ticket,
                (string)HistoryDealGetInteger(deal_ticket, DEAL_ORDER),
                (string)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID),
                symbol,
                DealTypeToText((ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE)),
                DealEntryToText((ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_VOLUME), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PRICE), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SL), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_TP), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PROFIT), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SWAP), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION), 2),
                comment);
     }

   FileClose(fh);
   return true;
  }

//+------------------------------------------------------------------+
//| Transaction Logger                                               |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(!InpEnableTransactionCsv)
      return;
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD || trans.deal == 0)
      return;

   AppendDealCsv(trans.deal);
  }
