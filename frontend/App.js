import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const App = () => {
  const [ppgData, setPpgData] = useState(Array(1000).fill(0));
  const [ecgData, setEcgData] = useState(Array(1000).fill(0));
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ppg: ppgData, ecg: ecgData }),
      });
      const data = await response.json();
      setPrediction(data.prediction[0]);
    } catch (error) {
      console.error('Error:', error);
    }
    setIsLoading(false);
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'PPG and ECG Signals',
      },
    },
  };

  const chartData = {
    labels: Array.from({ length: 1000 }, (_, i) => i + 1),
    datasets: [
      {
        label: 'PPG',
        data: ppgData,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'ECG',
        data: ecgData,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
    ],
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-4xl bg-white rounded-lg shadow-md p-6">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-4">
          Blood Pressure Prediction Tool
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h2 className="text-xl font-semibold mb-2">Input Signals</h2>
            <div className="mb-4">
              <label className="block text-gray-700">PPG Data (comma-separated)</label>
              <textarea
                className="w-full h-24 p-2 border rounded"
                onChange={(e) => setPpgData(e.target.value.split(',').map(Number))}
              />
            </div>
            <div>
              <label className="block text-gray-700">ECG Data (comma-separated)</label>
              <textarea
                className="w-full h-24 p-2 border rounded"
                onChange={(e) => setEcgData(e.target.value.split(',').map(Number))}
              />
            </div>
            <button
              className="mt-4 w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600 disabled:bg-blue-300"
              onClick={handlePredict}
              disabled={isLoading}
            >
              {isLoading ? 'Predicting...' : 'Predict Blood Pressure'}
            </button>
          </div>
          <div>
            <h2 className="text-xl font-semibold mb-2">Signal Visualization</h2>
            <Line options={chartOptions} data={chartData} />
          </div>
        </div>
        {prediction && (
          <div className="mt-6 p-4 bg-green-100 rounded">
            <h2 className="text-xl font-semibold">Predicted Blood Pressure</h2>
            <p className="text-lg">{prediction.map(p => p.toFixed(2)).join(', ')}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
