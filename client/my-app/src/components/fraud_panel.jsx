import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Target, Clock, Database, BarChart3, AlertTriangle, CheckCircle, Wifi, WifiOff } from 'lucide-react';

export default function FraudDetectionDashboard() {
  // Connection states
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState('');
  const websocketRef = useRef(null);
  
  // Processing states from server
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('Idle');
  const [accuracy, setAccuracy] = useState('--');
  const [processingTime, setProcessingTime] = useState('--');
  const [datasetsProcessed, setDatasetsProcessed] = useState('--');
  const [totalTransactions, setTotalTransactions] = useState('--');
  const [fraudDetected, setFraudDetected] = useState('--');
  const [fraudRate, setFraudRate] = useState('--');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [lastUpdate, setLastUpdate] = useState('--:--:--');

  // WebSocket connection and message handling
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        // Connect to your FastAPI WebSocket
        const ws = new WebSocket('ws://localhost:8000/ws');
        websocketRef.current = ws;

        ws.onopen = () => {
          console.log('✅ WebSocket connected');
          setIsConnected(true);
          setConnectionError('');
          
          // Request current status when connected
          ws.send(JSON.stringify({ command: 'get_status' }));
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('📨 Received data:', data);
            
            // Update all metrics from server
            if (data.accuracy !== undefined) setAccuracy(data.accuracy);
            if (data.processing_time !== undefined) setProcessingTime(data.processing_time);
            if (data.datasets_processed !== undefined) setDatasetsProcessed(data.datasets_processed);
            if (data.total_transactions !== undefined) setTotalTransactions(data.total_transactions);
            if (data.fraud_detected !== undefined) setFraudDetected(data.fraud_detected);
            if (data.fraud_rate !== undefined) setFraudRate(data.fraud_rate);
            if (data.status !== undefined) setStatus(data.status);
            if (data.is_processing !== undefined) setIsProcessing(data.is_processing);
            if (data.model_loaded !== undefined) setModelLoaded(data.model_loaded);
            if (data.last_updated !== undefined) {
              const updateTime = new Date(data.last_updated).toLocaleTimeString();
              setLastUpdate(updateTime);
            }
            
            // Handle errors
            if (data.error) {
              console.error('❌ Server error:', data.error);
              setConnectionError(data.error);
            }
          } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
          }
        };

        ws.onclose = () => {
          console.log('🔌 WebSocket disconnected');
          setIsConnected(false);
          setConnectionError('Disconnected from server');
          
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          setIsConnected(false);
          setConnectionError('Connection failed - make sure server is running on localhost:8000');
        };

      } catch (error) {
        console.error('❌ Failed to create WebSocket:', error);
        setConnectionError('Failed to connect to server');
      }
    };

    connectWebSocket();

    // Cleanup on unmount
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  // Send command to server
  const sendCommand = (command) => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      console.log(`📤 Sending command: ${command}`);
      websocketRef.current.send(JSON.stringify({ command }));
    } else {
      console.error('❌ WebSocket not connected');
      setConnectionError('Not connected to server');
    }
  };

  const handleStartProcessing = () => {
    sendCommand('start_processing');
  };

  const handleStopProcessing = () => {
    sendCommand('stop_processing');
  };

  const handleResetMetrics = () => {
    sendCommand('reset_metrics');
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        
        {/* Connection Status */}
        <div className={`rounded-lg p-4 ${
          isConnected 
            ? 'bg-green-50 border border-green-200' 
            : 'bg-red-50 border border-red-200'
        }`}>
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              {isConnected ? (
                <Wifi size={20} className="text-green-600 mt-0.5" />
              ) : (
                <WifiOff size={20} className="text-red-600 mt-0.5" />
              )}
            </div>
            <div>
              <h3 className={`text-sm font-medium ${
                isConnected ? 'text-green-800' : 'text-red-800'
              }`}>
                {isConnected ? 'Connected to Server' : 'Server Connection'}
              </h3>
              <p className={`text-sm mt-1 ${
                isConnected ? 'text-green-700' : 'text-red-700'
              }`}>
                {isConnected 
                  ? `Real-time connection established • Model: ${modelLoaded ? 'Loaded' : 'Not Loaded'}`
                  : connectionError || 'Attempting to connect to ws://localhost:8000/ws'
                }
              </p>
            </div>
          </div>
        </div>

        {/* Header Controls */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  status === 'Idle' ? 'bg-green-500' : 
                  status === 'Processing' ? 'bg-blue-500 animate-pulse' : 'bg-gray-400'
                }`}></div>
                <span className="text-sm font-medium text-gray-700">{status}</span>
              </div>
              {modelLoaded && (
                <div className="flex items-center space-x-1 text-green-600">
                  <CheckCircle size={14} />
                  <span className="text-xs">Model Ready</span>
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={handleStartProcessing}
                disabled={!isConnected || isProcessing}
                className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                <Play size={16} />
                <span>Start Processing</span>
              </button>
              
              <button
                onClick={handleStopProcessing}
                disabled={!isConnected || !isProcessing}
                className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                <Square size={16} />
                <span>Stop</span>
              </button>

              <button
                onClick={handleResetMetrics}
                disabled={!isConnected}
                className="flex items-center space-x-2 bg-orange-600 hover:bg-orange-700 disabled:bg-orange-400 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                <span>Reset</span>
              </button>
            </div>
          </div>
        </div>

        {/* Model Output Diagram */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <BarChart3 size={20} className="text-blue-600" />
                <h2 className="text-lg font-semibold text-gray-800">Real-time Fraud Detection</h2>
              </div>
              <span className="text-sm text-gray-500">Last update: {lastUpdate}</span>
            </div>
          </div>
          
          <div className="p-6">
            <div className="h-80 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200 flex items-center justify-center">
              {isProcessing ? (
                <div className="text-center space-y-4">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                  <p className="text-gray-600">Processing fraud detection model...</p>
                  <div className="bg-white rounded-lg p-4 shadow-sm">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">Transactions:</span>
                        <span className="ml-2 font-mono text-blue-600">{totalTransactions}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Fraud Detected:</span>
                        <span className="ml-2 font-mono text-red-600">{fraudDetected}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Fraud Rate:</span>
                        <span className="ml-2 font-mono text-orange-600">{fraudRate}%</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Accuracy:</span>
                        <span className="ml-2 font-mono text-green-600">{accuracy}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500">
                  <BarChart3 size={48} className="mx-auto mb-4 text-gray-300" />
                  <p className="text-lg font-medium">No active processing</p>
                  <p className="text-sm">Start processing to view real-time fraud detection</p>
                  {!isConnected && (
                    <p className="text-sm text-red-500 mt-2">
                      ⚠️ Connect to server first
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          
          {/* Accuracy */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Target size={20} className="text-blue-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600">Accuracy</p>
                <p className="text-2xl font-bold text-gray-900">
                  {accuracy}{accuracy !== '--' ? '%' : ''}
                </p>
              </div>
            </div>
          </div>

          {/* Processing Time */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <Clock size={20} className="text-green-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Time</p>
                <p className="text-2xl font-bold text-gray-900">
                  {processingTime}{processingTime !== '--' ? ' ms' : ''}
                </p>
              </div>
            </div>
          </div>

          {/* Total Transactions */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Database size={20} className="text-purple-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600">Transactions</p>
                <p className="text-2xl font-bold text-gray-900">{totalTransactions}</p>
              </div>
            </div>
          </div>

          {/* Fraud Detected */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-red-100 rounded-lg">
                <AlertTriangle size={20} className="text-red-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600">Fraud Detected</p>
                <p className="text-2xl font-bold text-gray-900">{fraudDetected}</p>
                <p className="text-xs text-gray-500">
                  {fraudRate !== '--' ? `${fraudRate}% rate` : ''}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Status Information */}
        {isProcessing && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse mt-2"></div>
              </div>
              <div>
                <h3 className="text-sm font-medium text-blue-800">Live Processing Active</h3>
                <p className="text-sm text-blue-700 mt-1">
                  Real-time fraud detection is running. The system is analyzing transactions and 
                  updating metrics every 2 seconds. {totalTransactions !== '--' ? 
                  `Processed ${totalTransactions} transactions so far.` : ''}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* No Data Warning */}
        {isConnected && !isProcessing && totalTransactions === 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <AlertTriangle size={20} className="text-yellow-600 mt-0.5" />
              </div>
              <div>
                <h3 className="text-sm font-medium text-yellow-800">No Transaction Data</h3>
                <p className="text-sm text-yellow-700 mt-1">
                  Your database appears to be empty (0 transactions found). 
                  The system will use mock data for testing when you start processing.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}