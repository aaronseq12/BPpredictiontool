/**
 * Advanced Blood Pressure Prediction Tool - Main Application
 * Simplified version using existing components
 */
import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';

// Components
import PredictionInterface from './components/Prediction/PredictionInterface';

// Styles
import './index.css';

const App = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [apiHealth, setApiHealth] = useState(null);

  // Check API health on app load
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
        const response = await fetch(`${apiUrl}/health`);
        const healthData = await response.json();
        setApiHealth(healthData);
      } catch (error) {
        console.error('Failed to check API health:', error);
        setApiHealth({ status: 'unhealthy', error: error.message });
      } finally {
        setIsLoading(false);
      }
    };

    checkApiHealth();
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Loading Blood Pressure Prediction Tool...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-indigo-600">
              Blood Pressure Prediction Tool
            </h1>
            <div className="flex items-center space-x-2">
              <span className={`inline-block w-3 h-3 rounded-full ${
                apiHealth?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
              }`}></span>
              <span className="text-sm text-gray-600">
                API: {apiHealth?.status === 'healthy' ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <PredictionInterface />
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-auto">
        <div className="container mx-auto px-4 py-4 text-center text-gray-600">
          <p>&copy; 2024 Blood Pressure Prediction Tool. Built with AI/ML.</p>
        </div>
      </footer>

      {/* Toast notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            style: {
              background: '#10b981',
            },
          },
          error: {
            style: {
              background: '#ef4444',
            },
          },
        }}
      />
    </div>
  );
};

export default App;
