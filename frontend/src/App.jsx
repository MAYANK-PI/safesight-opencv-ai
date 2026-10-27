import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import Home from './components/Home';
import Incidents from './components/Incidents';

function App() {
  const [tab, setTab] = useState('home'); // 'home' or 'incidents'

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Industrial Safety Inspector</h1>
        <div className="tab-bar">
          <button
            className={tab === 'home' ? 'active' : ''}
            onClick={() => setTab('home')}
          >
            Home
          </button>
          <button
            className={tab === 'incidents' ? 'active' : ''}
            onClick={() => setTab('incidents')}
          >
            Incidents
          </button>
        </div>
      </header>
      <main>
        {tab === 'home' ? <Home /> : <Incidents />}
      </main>
    </div>
  );
}

export default App;