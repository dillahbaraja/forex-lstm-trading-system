//+------------------------------------------------------------------+
//|                                                 DataExporter.mq5 |
//|                                  Copyright 2026, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Antigravity AI"
#property link      "https://www.mql5.com"
#property version   "1.00"

// --- Inputs ---
input int InpHistoryBars = 35000;    // Number of historical bars to export (~5 years on H1)
input string InpTargetSymbol = "";   // Symbol to export, leave empty to use chart symbol
input bool InpAutoFileName = true;   // Auto-generate file name as <SYMBOL>_<TF>_Data.csv
input string InpFileName = "";       // Custom output file name when auto name is disabled
input int InpRsiPeriod = 14;        // RSI Period
input int InpAtrPeriod = 14;        // ATR Period

int handleRSI, handleATR, handleMACD, handleBands;
int handleStoch, handleADX; // NEW: Stochastic & ADX
int handleRSI_H4, handleMA_H4, handleRSI_M15, handleATR_M15;
string gTargetSymbol = "";
string gOutputFileName = "";

string ResolveTargetSymbol();
string ResolveOutputFileName(const string symbol);
string TimeframeToLabel(const ENUM_TIMEFRAMES timeframe);
string Trimmed(const string value);

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   gTargetSymbol = ResolveTargetSymbol();
   if(!SymbolSelect(gTargetSymbol, true))
     {
      Print("Failed to select target symbol: ", gTargetSymbol, ". Err: ", GetLastError());
      return(INIT_FAILED);
     }

   gOutputFileName = ResolveOutputFileName(gTargetSymbol);

   // Initialize indicator handles
   handleRSI = iRSI(gTargetSymbol, _Period, InpRsiPeriod, PRICE_CLOSE);
   handleATR = iATR(gTargetSymbol, _Period, InpAtrPeriod);
   handleMACD = iMACD(gTargetSymbol, _Period, 12, 26, 9, PRICE_CLOSE);
   handleBands = iBands(gTargetSymbol, _Period, 20, 0, 2.0, PRICE_CLOSE);
   handleStoch = iStochastic(gTargetSymbol, _Period, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   handleADX = iADX(gTargetSymbol, _Period, 14);
   
   // MTF Handles
   handleRSI_H4 = iRSI(gTargetSymbol, PERIOD_H4, InpRsiPeriod, PRICE_CLOSE);
   handleMA_H4 = iMA(gTargetSymbol, PERIOD_H4, 24, 0, MODE_SMA, PRICE_CLOSE);
   handleRSI_M15 = iRSI(gTargetSymbol, PERIOD_M15, InpRsiPeriod, PRICE_CLOSE);
   handleATR_M15 = iATR(gTargetSymbol, PERIOD_M15, InpAtrPeriod);
   
   if(handleRSI == INVALID_HANDLE || handleATR == INVALID_HANDLE || 
      handleMACD == INVALID_HANDLE || handleBands == INVALID_HANDLE ||
      handleStoch == INVALID_HANDLE || handleADX == INVALID_HANDLE ||
      handleRSI_H4 == INVALID_HANDLE || handleMA_H4 == INVALID_HANDLE ||
      handleRSI_M15 == INVALID_HANDLE || handleATR_M15 == INVALID_HANDLE)
     {
      Print("Failed to create indicator handles. Err: ", GetLastError());
      return(INIT_FAILED);
     }
     
   Print("Initialization successful. Exporting data...");
   ExportData();
   
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Export Data Function                                             |
//+------------------------------------------------------------------+
void ExportData()
  {
   int fileHandle = FileOpen(gOutputFileName, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   
   if(fileHandle == INVALID_HANDLE)
     {
      Print("Error opening file: ", GetLastError());
      return;
     }
     
   // Write Header
   FileWrite(fileHandle, "Time", "Open", "High", "Low", "Close", "Volume", "Hour", "DayOfWeek", "RSI", "ATR", "MACD_Main", "MACD_Signal", "Bands_Upper", "Bands_Lower", "Stoch_Main", "Stoch_Signal", "ADX_Main", "ADX_PlusDI", "ADX_MinusDI", "RSI_H4", "MA_H4", "RSI_M15", "ATR_M15");

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   double rsiBuffer[], atrBuffer[], macdMain[], macdSignal[], bandsUpper[], bandsLower[];
   double stochMain[], stochSignal[], adxMain[], adxPlus[], adxMinus[];
   
   ArraySetAsSeries(rsiBuffer, true);
   ArraySetAsSeries(atrBuffer, true);
   ArraySetAsSeries(macdMain, true);
   ArraySetAsSeries(macdSignal, true);
   ArraySetAsSeries(bandsUpper, true);
   ArraySetAsSeries(bandsLower, true);
   ArraySetAsSeries(stochMain, true);
   ArraySetAsSeries(stochSignal, true);
   ArraySetAsSeries(adxMain, true);
   ArraySetAsSeries(adxPlus, true);
   ArraySetAsSeries(adxMinus, true);

   // Copy data
   int copied = CopyRates(gTargetSymbol, _Period, 0, InpHistoryBars, rates);
   CopyBuffer(handleRSI, 0, 0, InpHistoryBars, rsiBuffer);
   CopyBuffer(handleATR, 0, 0, InpHistoryBars, atrBuffer);
   CopyBuffer(handleMACD, 0, 0, InpHistoryBars, macdMain);
   CopyBuffer(handleMACD, 1, 0, InpHistoryBars, macdSignal);
   CopyBuffer(handleBands, 1, 0, InpHistoryBars, bandsUpper); // Upper band
   CopyBuffer(handleBands, 2, 0, InpHistoryBars, bandsLower); // Lower band
   CopyBuffer(handleStoch, 0, 0, InpHistoryBars, stochMain);
   CopyBuffer(handleStoch, 1, 0, InpHistoryBars, stochSignal);
   CopyBuffer(handleADX, 0, 0, InpHistoryBars, adxMain);
   CopyBuffer(handleADX, 1, 0, InpHistoryBars, adxPlus);
   CopyBuffer(handleADX, 2, 0, InpHistoryBars, adxMinus);
   
   if(copied <= 0)
     {
      Print("Failed to copy rates.");
      FileClose(fileHandle);
      return;
     }

   // Write data to CSV
   for(int i = copied - 1; i >= 0; i--)
     {
      int h4Shift = iBarShift(gTargetSymbol, PERIOD_H4, rates[i].time, false);
      int m15Shift = iBarShift(gTargetSymbol, PERIOD_M15, rates[i].time, false);
      if(h4Shift < 0 || m15Shift < 0)
        {
         Print("Failed to resolve MTF shift for bar time: ", TimeToString(rates[i].time));
         FileClose(fileHandle);
         return;
        }

      double rsi_h4[1], ma_h4[1], rsi_m15[1], atr_m15[1];
      if(CopyBuffer(handleRSI_H4, 0, h4Shift, 1, rsi_h4) < 1 ||
         CopyBuffer(handleMA_H4, 0, h4Shift, 1, ma_h4) < 1 ||
         CopyBuffer(handleRSI_M15, 0, m15Shift, 1, rsi_m15) < 1 ||
         CopyBuffer(handleATR_M15, 0, m15Shift, 1, atr_m15) < 1)
        {
         Print("Failed to copy MTF indicator values for bar time: ", TimeToString(rates[i].time));
         FileClose(fileHandle);
         return;
        }
      
      MqlDateTime dt;
      TimeToStruct(rates[i].time, dt);
      
      int targetDigits = (int)SymbolInfoInteger(gTargetSymbol, SYMBOL_DIGITS);
      FileWrite(fileHandle, 
                TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
                DoubleToString(rates[i].open, targetDigits),
                DoubleToString(rates[i].high, targetDigits),
                DoubleToString(rates[i].low, targetDigits),
                DoubleToString(rates[i].close, targetDigits),
                IntegerToString(rates[i].tick_volume),
                IntegerToString(dt.hour),
                IntegerToString(dt.day_of_week),
                DoubleToString(rsiBuffer[i], 2),
                DoubleToString(atrBuffer[i], targetDigits),
                DoubleToString(macdMain[i], targetDigits),
                DoubleToString(macdSignal[i], targetDigits),
                DoubleToString(bandsUpper[i], targetDigits),
                DoubleToString(bandsLower[i], targetDigits),
                DoubleToString(stochMain[i], 2),
                DoubleToString(stochSignal[i], 2),
                DoubleToString(adxMain[i], 2),
                DoubleToString(adxPlus[i], 2),
                DoubleToString(adxMinus[i], 2),
                DoubleToString(rsi_h4[0], 2),
                DoubleToString(ma_h4[0], targetDigits),
                DoubleToString(rsi_m15[0], 2),
                DoubleToString(atr_m15[0], targetDigits)
               );
     }
     
   FileClose(fileHandle);
   Print("Data exported successfully to: ", gOutputFileName, " for symbol ", gTargetSymbol);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   IndicatorRelease(handleRSI);
   IndicatorRelease(handleATR);
   IndicatorRelease(handleMACD);
   IndicatorRelease(handleBands);
   IndicatorRelease(handleStoch);
   IndicatorRelease(handleADX);
   IndicatorRelease(handleRSI_H4);
   IndicatorRelease(handleMA_H4);
   IndicatorRelease(handleRSI_M15);
   IndicatorRelease(handleATR_M15);
  }

string ResolveTargetSymbol()
  {
   string symbol = Trimmed(InpTargetSymbol);
   if(StringLen(symbol) == 0)
      return _Symbol;
   StringToUpper(symbol);
   return symbol;
  }

string ResolveOutputFileName(const string symbol)
  {
   string fileName = Trimmed(InpFileName);
   if(InpAutoFileName || StringLen(fileName) == 0)
      return symbol + "_" + TimeframeToLabel(_Period) + "_Data.csv";
   return fileName;
  }

string TimeframeToLabel(const ENUM_TIMEFRAMES timeframe)
  {
   switch(timeframe)
     {
      case PERIOD_M1: return "M1";
      case PERIOD_M5: return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1: return "H1";
      case PERIOD_H4: return "H4";
      case PERIOD_D1: return "D1";
      case PERIOD_W1: return "W1";
      case PERIOD_MN1: return "MN1";
      default: return IntegerToString((int)timeframe);
     }
  }

string Trimmed(const string value)
  {
   string trimmed = value;
   StringTrimLeft(trimmed);
   StringTrimRight(trimmed);
   return trimmed;
  }
