import React, {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {MapContainer, TileLayer, CircleMarker, Popup, ImageOverlay, Polyline, Circle, useMap} from 'react-leaflet';
import L from 'leaflet';
import {Activity, ArrowRight, Bot, CheckCircle2, ChevronDown, FileText, Gauge, Globe2, Info, LogIn, MapPin, Radio, RefreshCw, Satellite, ShieldCheck, Waves, Wind} from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import './styles.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const INSAT_SOURCE = 'https://mausam.imd.gov.in/responsive/satellite_rapidscan.php';
const IMD_SOURCE = 'https://mausam.imd.gov.in/';
const SATELLITE_PRODUCTS = {
  'VIS': ['https://mausam.imd.gov.in/Satellite/rswmo_vis.jpg'],
  'IR-1': ['https://mausam.imd.gov.in/Satellite/rswmo_ir1.jpg'],
  'SWIR': ['https://mausam.imd.gov.in/Satellite/rswmo_swir.jpg'],
  'WV': ['https://mausam.imd.gov.in/Satellite/rswmo_wv.jpg'],
  'MP': ['https://mausam.imd.gov.in/Satellite/rswmo_mp.jpg'],
  'BD Curve': [
    'https://mausam.imd.gov.in/Satellite/rswmo_bdcurve.jpg',
    'https://mausam.imd.gov.in/Satellite/rswmo_bd_curve.jpg'
  ],
  'NHC Curve': [
    'https://mausam.imd.gov.in/Satellite/rswmo_nhccurve.jpg',
    'https://mausam.imd.gov.in/Satellite/rswmo_nhc_curve.jpg'
  ],
  'OLR': ['https://mausam.imd.gov.in/Satellite/rswmo_olr.jpg'],
  'HEM': [
    'https://mausam.imd.gov.in/Satellite/rswmo_hem.jpg',
    'https://mausam.imd.gov.in/Satellite/rswmo_he.jpg'
  ],
  'IMR': ['https://mausam.imd.gov.in/Satellite/rswmo_imr.jpg']
};
const SATELLITE_SECTORS = {
  'Asia Sector': ['VIS','IR-1','SWIR','WV','MP','BD Curve','NHC Curve'],
  'Arabian Sea': ['VIS','IR-1','SWIR','WV','MP','BD Curve','NHC Curve'],
  'Bay of Bengal': ['VIS','IR-1','SWIR','WV','MP','BD Curve','NHC Curve'],
  'Geophysical': ['OLR','HEM','IMR']
};
const SATELLITE_INFO = {
  'VIS':'Visible imagery · cloud structure and daytime convection',
  'IR-1':'Infrared-1 · cloud-top temperature / convection structure',
  'SWIR':'Short-wave infrared · low-level cloud / night-time structure',
  'WV':'Water vapour · moisture and mid/upper-level circulation',
  'MP':'Microwave product · precipitation / cloud-structure context',
  'BD Curve':'BD Curve diagnostic product · official IMD Rapid Scan',
  'NHC Curve':'NHC Curve cyclone-analysis product · official IMD Rapid Scan',
  'OLR':'Outgoing Longwave Radiation · convection / cloud-top context',
  'HEM':'Hydro-Estimator Method rainfall product',
  'IMR':'INSAT Multi-spectral Rainfall product'
};
const REGIONS = [
 ['kochi','Kochi Coast','Kerala','Arabian Sea',9.9312,76.2673],
 ['mangaluru','Mangaluru Coast','Karnataka','Arabian Sea',12.9141,74.8560],
 ['goa','Goa Coast','Goa','Arabian Sea',15.4909,73.8278],
 ['mumbai','Mumbai Coast','Maharashtra','Arabian Sea',19.0760,72.8777],
 ['konkan','Konkan Coast','Maharashtra','Arabian Sea',16.9902,73.3120],
 ['gujarat','Gujarat Coast','Gujarat','Arabian Sea',21.6417,69.6293],
 ['kutch','Kutch Coast','Gujarat','Arabian Sea',23.7337,69.8597],
 ['lakshadweep','Lakshadweep','Lakshadweep','Arabian Sea',10.5667,72.6417],
 ['south_arabian','South Arabian Sea','—','Arabian Sea',12.0000,65.0000],
 ['north_arabian','North Arabian Sea','—','Arabian Sea',20.0000,64.0000],
 ['chennai','Chennai Coast','Tamil Nadu','Bay of Bengal',13.0827,80.2707],
 ['puducherry','Puducherry Coast','Puducherry','Bay of Bengal',11.9416,79.8083],
 ['andhra','Andhra Coast','Andhra Pradesh','Bay of Bengal',16.0000,81.0000],
 ['visakhapatnam','Visakhapatnam Coast','Andhra Pradesh','Bay of Bengal',17.6868,83.2185],
 ['paradip','Paradip Coast','Odisha','Bay of Bengal',20.3167,86.6083],
 ['odisha','Odisha Coast','Odisha','Bay of Bengal',19.8135,85.8312],
 ['west_bengal','West Bengal Coast','West Bengal','Bay of Bengal',21.9497,87.7479],
 ['north_bob','North Bay of Bengal','West Bengal','Bay of Bengal',20.8000,89.5000],
 ['central_bob','Central Bay of Bengal','—','Bay of Bengal',15.5000,88.5000],
 ['andaman','Andaman Sea','Andaman & Nicobar Islands','Bay of Bengal',11.7401,92.6586],
].map(([id,name,state,basin,lat,lon])=>({id,name,state,basin,lat,lon}));

const CLASS_NAMES = ['Depression / Weak System','Cyclonic Storm','Severe Cyclone'];


function Vortex(){return <div className="vortex" aria-hidden="true"><div className="vortex-core">X</div><i/><i/><i/><i/><span className="vortex-cloud c1"/><span className="vortex-cloud c2"/><span className="vortex-cloud c3"/></div>}
function Reveal({children,className=''}){return <div className={`reveal ${className}`}>{children}</div>}
function MapReset({selected}){const map=useMap(); useEffect(()=>{const r=REGIONS.find(x=>x.id===selected); if(r) map.flyTo([r.lat,r.lon],5.2,{duration:.7});},[selected]); return null}
function IndiaMap({selected,setSelected,showSatellite,forecast,satelliteLayer,liveTick}){
 return <div className="map-shell">
  <MapContainer center={[18.5,79]} zoom={4.6} minZoom={3.7} maxZoom={8} scrollWheelZoom className="india-map" zoomControl={false}>
   <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
   {showSatellite && SATELLITE_PRODUCTS[satelliteLayer]?.[0] && ['VIS','IR-1','SWIR','WV','MP'].includes(satelliteLayer) && <ImageOverlay key={satelliteLayer} url={`${SATELLITE_PRODUCTS[satelliteLayer][0]}?t=${liveTick}`} bounds={[[-10,50],[40,110]]} opacity={.62} zIndex={300} />}
   <MapReset selected={selected}/>
   {REGIONS.map(r=><CircleMarker key={r.id} center={[r.lat,r.lon]} radius={r.id===selected?9:6} pathOptions={{color:r.id===selected?'#00b8d9':'#147c96',fillColor:r.id===selected?'#dffaff':'#37a9c4',fillOpacity:.95,weight:2}} eventHandlers={{click:()=>setSelected(r.id)}}><Popup><b>{r.name}</b><br/>{r.basin}<br/>{r.lat.toFixed(3)}°N · {r.lon.toFixed(3)}°E</Popup></CircleMarker>)}
   {forecast?.forecast?.length>0 && <><CircleMarker center={[forecast.origin_lat,forecast.origin_lon]} radius={10} pathOptions={{color:'#ffcc66',fillColor:'#ffcc66',fillOpacity:.9,weight:3}}/><Polyline positions={[[forecast.origin_lat,forecast.origin_lon],...forecast.forecast.map(f=>[f.lat,f.lon])]} pathOptions={{color:'#00e5ff',weight:4,dashArray:'7 7'}}/>{forecast.forecast.map(f=><Circle key={f.horizon_h} center={[f.lat,f.lon]} radius={f.uncertainty_km*1000} pathOptions={{color:'#00e5ff',fillColor:'#00e5ff',fillOpacity:.035,weight:1,dashArray:'4 6'}}/>)}</>}
  </MapContainer>
  <div className="map-toolbar"><span className="map-pill"><span className="pulse-dot"/> INDIA / NORTH INDIAN OCEAN</span><span className="map-pill">{showSatellite?`INSAT-3DR ${satelliteLayer}`:'BASE MAP'}</span></div>
 </div>
}

function SatelliteProductViewer({product,sector}){
 const urls=SATELLITE_PRODUCTS[product]||[];
 const [idx,setIdx]=useState(0);
 const [failed,setFailed]=useState(false);
 useEffect(()=>{setIdx(0);setFailed(false)},[product,sector]);
 const src=urls[idx];
 return <div className="sat-product-viewer">
   <div className="sat-viewer-head"><div><span className="kicker">OFFICIAL PRODUCT VIEW</span><h3>{sector} · {product}</h3></div><span className="live-chip"><span className="pulse-dot"/> IMD SOURCE</span></div>
   {!failed && src ? <div className="sat-image-frame"><img src={`${src}?t=${Date.now()}`} alt={`${sector} ${product} INSAT-3DR`} onError={()=>{if(idx+1<urls.length)setIdx(idx+1);else setFailed(true)}}/><div className="sat-image-caption">INSAT-3DR · India Meteorological Department · refreshed on selection</div></div> : <div className="sat-fallback"><Satellite size={28}/><b>{product} is available through the official IMD Rapid Scan service.</b><span>The direct image endpoint is not exposed consistently for every derived product, so CYCLONE-X opens the authoritative product service instead of fabricating an image.</span><a href={INSAT_SOURCE} target="_blank" rel="noreferrer">Open {product} in official IMD Rapid Scan ↗</a></div>}
 </div>
}

function App(){
 const [page,setPage]=useState(localStorage.getItem('cx_logged')==='1'?'dashboard':'landing');
 const [logged,setLogged]=useState(localStorage.getItem('cx_logged')==='1');
 const [user,setUser]=useState(localStorage.getItem('cx_user')||'Operator'); const [email,setEmail]=useState(''); const [forecast,setForecast]=useState(null); const [forecastLoading,setForecastLoading]=useState(false); const [password,setPassword]=useState(''); const [loginError,setLoginError]=useState('');
 const [selected,setSelected]=useState('kochi'); const [satelliteSector,setSatelliteSector]=useState('Arabian Sea'); const [satelliteLayer,setSatelliteLayer]=useState('VIS'); const [analysis,setAnalysis]=useState(null); const [dailyWeather,setDailyWeather]=useState(null); const [weatherLoading,setWeatherLoading]=useState(false); const [backend,setBackend]=useState(false); const [loading,setLoading]=useState(false); const [showSatellite,setShowSatellite]=useState(true); const [liveTick,setLiveTick]=useState(Date.now()); const [scenario,setScenario]=useState(''); const [agent,setAgent]=useState('');
 const region=useMemo(()=>REGIONS.find(r=>r.id===selected)||REGIONS[0],[selected]);
 useEffect(()=>{document.documentElement.dataset.theme='light';},[]);
 useEffect(()=>{fetch(`${API}/api/health`).then(r=>r.json()).then(x=>setBackend(Boolean(x.model_loaded))).catch(()=>setBackend(false));},[]);
 useEffect(()=>{const t=setInterval(()=>setLiveTick(Date.now()),300000);return()=>clearInterval(t)},[]);
 const currentStatus = analysis?.observation || {active_cyclone:false,status:'NO ACTIVE CYCLONIC STORM DETECTED',message:'No cyclone-stage event is being assigned to this region without an observation/model result.'};
 async function runAnalysis(stage=''){
  setLoading(true); setScenario(stage); setAgent('');
  try{const q=stage?`?scenario_stage=${encodeURIComponent(stage)}`:''; const r=await fetch(`${API}/api/analyze/${selected}${q}`); if(!r.ok) throw new Error('API error'); setAnalysis(await r.json());}
  catch(e){setAnalysis({region,observation:{active_cyclone:false,status:'BACKEND UNAVAILABLE',message:'Connect FastAPI before using model analysis.'},classification:{active:false,stage:'No active cyclone'},state_risk:REGIONS.map(()=>({zone:'GREEN'})),ai_agent:{summary:'Backend unavailable.',precautions:['Start FastAPI on port 8000.']},disclaimer:'Local interface only.'});}
  finally{setLoading(false)}
 }
 async function runForecast(){
  setForecastLoading(true);
  try{const r=await fetch(`${API}/api/forecast-demo/100`); if(!r.ok) throw new Error(await r.text()); setForecast(await r.json());}
  catch(e){setForecast({error:'Forecast model is not trained yet. Run train_cyclonex_forecast.py on the local dataset first.'});}
  finally{setForecastLoading(false)}
 }
 async function runDailyWeather(){
  setWeatherLoading(true);
  try{const r=await fetch(`${API}/api/weather-outlook/${selected}`); if(!r.ok) throw new Error(await r.text()); setDailyWeather(await r.json());}
  catch(e){setDailyWeather({error:'Daily weather service is unavailable. Check the FastAPI connection and internet access.'});}
  finally{setWeatherLoading(false)}
 }
 function login(e){e.preventDefault();if(email.trim().length<3||password.length<4){setLoginError('Enter a valid operator email and password.');return}localStorage.setItem('cx_logged','1');localStorage.setItem('cx_user',email.split('@')[0]);setLogged(true);setUser(email.split('@')[0]);setPage('dashboard');setLoginError('')}
 function logout(){localStorage.removeItem('cx_logged');localStorage.removeItem('cx_user');setLogged(false);setPage('landing')}
 const goLogin=()=>setPage('login');
 const scrollTo=(id)=>document.getElementById(id)?.scrollIntoView({behavior:'smooth'});
 return <div className="app">
  {page==='landing' && <header className="site-nav glass"><button className="brand" onClick={()=>scrollTo('top')}><span className="brand-mark"><Waves size={18}/></span><span>CYCLONE-X</span></button><nav><button onClick={()=>scrollTo('platform')}>Platform</button><button onClick={()=>scrollTo('architecture')}>Architecture</button><button onClick={()=>scrollTo('sources')}>Data</button></nav><div className="nav-right"><span className="status-chip"><span className="pulse-dot"/> INDIA / NIO</span><button className="nav-login" onClick={goLogin}>Enter console <ArrowRight size={15}/></button></div></header>}
  {page==='landing' && <main id="top">
   <section className="hero2 section-wide">
    <Reveal className="hero2-copy"><div className="eyebrow">SMART INDIA HACKATHON 2026 · PS 26070 · ARAMBH</div><h1>CYCLONE<span>-X</span></h1><div className="hero-line">Observe. Understand. Predict. Explain. Prepare.</div><p>India-first cyclone intelligence for the Arabian Sea and Bay of Bengal — combining satellite observation, historical cyclone tracks and environmental context into a transparent decision-support workflow.</p><div className="hero-actions"><button className="primary" onClick={goLogin}>Enter intelligence console <ArrowRight size={16}/></button><button className="ghost" onClick={()=>scrollTo('platform')}>Explore platform <ChevronDown size={16}/></button></div><div className="hero-meta"><span><ShieldCheck size={14}/> Official warnings remain authoritative.</span><span><Satellite size={14}/> INSAT-3DR observation layer</span><span><Activity size={14}/> Model 1 connected locally</span></div></Reveal>
    <Reveal className="hero2-art"><div className="hero-particles">{Array.from({length:22},(_,i)=><i key={i} style={{"--i":i}}/> )}</div><div className="art-halo"/><Vortex/><div className="art-card glass"><span>MISSION STATE</span><b>INDIAN COASTAL INTELLIGENCE</b><small>Satellite → model → explanation → readiness</small></div></Reveal>
   </section>
   <section id="platform" className="section-wide"><Reveal className="section-title"><div><span className="kicker">01 · PLATFORM</span><h2>A command layer that changes when the evidence changes.</h2></div><p>CYCLONE-X is designed around regional observation. Select a coastal location, inspect the live satellite layer, run the connected model and keep scenario outputs visibly separate from real observations.</p></Reveal><div className="process-grid">{[['01','OBSERVE','INSAT / satellite imagery'],['02','DETECT','Cyclone structure signals'],['03','CLASSIFY','Model 1 cyclone stage'],['04','FUSE','Track + environment'],['05','EXPLAIN','Evidence and uncertainty'],['06','PREPARE','Region-aware readiness']].map(x=><Reveal key={x[0]} className="process-card glass"><span>{x[0]}</span><b>{x[1]}</b><small>{x[2]}</small></Reveal>)}</div></section>
   <section id="architecture" className="section-wide split-section"><Reveal className="feature-large glass"><div className="feature-top"><span className="kicker">02 · INTELLIGENCE CORE</span><Satellite size={20}/></div><h3>Multi-source, India-first analysis.</h3><p>Historical HURSAT-B1 satellite observations, NOAA IBTrACS cyclone tracks and ERA5 environmental context support the trained Model 1 pipeline. The live operational satellite layer is shown separately so the prototype never confuses a live image with historical training data.</p><div className="mini-bars">{[48,72,54,84,64,92,76,88].map((h,i)=><i key={i} style={{height:`${h}%`}}/>)}</div></Reveal><div className="feature-stack"><Reveal className="feature glass"><MapPin/><b>20 Indian / NIO regions</b><small>Coastal cities, coasts and Arabian Sea / Bay of Bengal sectors.</small></Reveal><Reveal className="feature glass"><Bot/><b>Explainable AI layer</b><small>Shows what the system observed, what it classified and what remains uncertain.</small></Reveal><Reveal className="feature glass"><Gauge/><b>Scenario isolation</b><small>Storm simulations are visually marked as scenarios — never presented as live warnings.</small></Reveal></div></section>
   <section id="sources" className="section-wide"><Reveal className="section-title"><div><span className="kicker">03 · DATA FOUNDATION</span><h2>Indian observation + research datasets.</h2></div></Reveal><div className="source-grid2">{[['INSAT-3D / 3DR','Indian meteorological satellite observation','Live layer'],['NOAA HURSAT-B1','Storm-centred historical satellite data','Training / validation'],['NOAA IBTrACS','Historical cyclone tracks','Track context'],['ERA5','Atmospheric environmental context','Fusion context']].map(x=><div className="source-card glass" key={x[0]}><span>{x[2]}</span><b>{x[0]}</b><p>{x[1]}</p></div>)}</div></section>
   <footer className="site-footer">CYCLONE-X · TEAM ARAMBH · SMART INDIA HACKATHON 2026 · PROTOTYPE / RESEARCH BUILD</footer>
  </main>}

  {page==='login' && <main className="auth-page"><div className="auth-art"><div className="auth-brand"><span className="brand-mark"><Waves size={18}/></span>CYCLONE-X</div><span className="kicker">INDIA / NORTH INDIAN OCEAN</span><h1>Enter the intelligence console.</h1><p>Monitor Indian coastal regions, inspect the INSAT-3DR observation layer and run the connected Model 1 pipeline.</p><div className="auth-vortex"><Vortex/></div></div><form className="auth-card glass" onSubmit={login}><div className="auth-tabs"><span className="active">OPERATOR LOGIN</span><span>LOCAL PROTOTYPE</span></div><h2>Welcome back.</h2><p>Use a local prototype credential. No external identity provider is used.</p><label>Email<input value={email} onChange={e=>setEmail(e.target.value)} placeholder="operator@cyclonex.local" autoComplete="username"/></label><label>Password<input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="••••••••" autoComplete="current-password"/></label>{loginError&&<div className="form-error">{loginError}</div>}<button className="primary full" type="submit"><LogIn size={16}/> Enter command center</button><button type="button" className="back-link" onClick={()=>setPage('landing')}>← Back to CYCLONE-X</button></form></main>}

  {page==='dashboard' && <main className="console">
   <header className="console-nav glass"><button className="brand" onClick={()=>setPage('dashboard')}><span className="brand-mark"><Waves size={17}/></span>CYCLONE-X</button><div className="console-right"><span className="status-chip"><span className="pulse-dot"/> LIVE LOOP · {new Date(liveTick).toLocaleTimeString()}</span><span className="operator">{user}</span><button className="nav-login" onClick={logout}>Log out</button></div></header>
   <section className="console-head"><div><span className="kicker">COMMAND CENTER · LOCATION-FIRST INTELLIGENCE</span><h1>Select an Indian coastal region.</h1><p>The live observation layer and model result are kept separate. Run analysis only when you want to evaluate the selected region.</p></div><div className="run-box glass"><label>MONITORING REGION<select value={selected} onChange={e=>{setSelected(e.target.value);setAnalysis(null);setScenario('');setDailyWeather(null);setForecast(null)}}>{REGIONS.map(r=><option key={r.id} value={r.id}>{r.name} · {r.basin}</option>)}</select></label><div className="run-actions"><button className="primary" disabled={loading} onClick={()=>runAnalysis('')}>{loading?<RefreshCw className="spin"/>:<Activity size={16}/>} {loading?'Analyzing…':'Run AI analysis'}</button><button className="ghost" onClick={()=>runAnalysis('Cyclonic Storm')}>Scenario</button><button className="ghost" onClick={runDailyWeather}>7-day weather</button></div></div></section>
   <section className="console-grid top-grid"><div className="panel-xl glass"><div className="panel-head"><div><span className="kicker">01 · LIVE OBSERVATION</span><h2>India monitoring map</h2></div><div className="toggle-group"><button className={showSatellite?'selected':''} onClick={()=>setShowSatellite(true)}><Satellite size={14}/> INSAT-3DR</button><button className={!showSatellite?'selected':''} onClick={()=>setShowSatellite(false)}>Base map</button></div></div><div className="satellite-layer-panel"><div className="satellite-sector-tabs">{Object.keys(SATELLITE_SECTORS).map(sec=><button key={sec} className={satelliteSector===sec?'active':''} onClick={()=>{setSatelliteSector(sec);setSatelliteLayer(SATELLITE_SECTORS[sec][0])}}>{sec}<span>+</span></button>)}</div><div className="satellite-products">{SATELLITE_SECTORS[satelliteSector].map(product=><button key={product} className={satelliteLayer===product?'active':''} onClick={()=>{setSatelliteLayer(product);setShowSatellite(true)}}>{product}</button>)}</div><SatelliteProductViewer product={satelliteLayer} sector={satelliteSector}/><div className="satellite-product-meta"><div><Satellite size={15}/><b>{satelliteSector} · {satelliteLayer}</b></div><span>{SATELLITE_INFO[satelliteLayer]}</span><a href={INSAT_SOURCE} target="_blank" rel="noreferrer">Open official IMD Rapid Scan ↗</a></div></div><IndiaMap selected={selected} setSelected={id=>{setSelected(id);setAnalysis(null);setDailyWeather(null)}} showSatellite={showSatellite} forecast={forecast} satelliteLayer={satelliteLayer} liveTick={liveTick}/><div className="source-note"><span><span className="pulse-dot"/> Official IMD INSAT-3DR Rapid Scan layer · {satelliteSector} / {satelliteLayer}</span><a href={INSAT_SOURCE} target="_blank" rel="noreferrer">Open IMD rapid scan ↗</a></div></div>
    <div className="panel-xl glass region-panel"><div className="panel-head"><span className="kicker">SELECTED REGION</span><MapPin size={18}/></div><h2>{region.name}</h2><p>{region.state} · {region.basin}</p><div className="coords">{region.lat.toFixed(3)}° N · {region.lon.toFixed(3)}° E</div><div className={`live-status ${currentStatus.active_cyclone?'active':''}`}><span className="status-dot"/>{currentStatus.active_cyclone?currentStatus.status:'NO ACTIVE CYCLONE DETECTED'}</div><div className="source-box"><div><small>Satellite observation</small><b>INSAT-3DR / IMD</b><span>Live official image</span></div><div><small>Model 1</small><b>{backend?'CONNECTED':'OFFLINE'}</b><span>{analysis?.classification?.stage||'Awaiting analysis'}</span></div></div><div className="region-note"><Info size={14}/><span>Real mode does not invent a storm. Scenario mode is clearly labelled and can be used for the jury demonstration.</span></div></div></section>
   <section className="console-grid lower-grid"><div className="panel-xl glass"><div className="panel-head"><div><span className="kicker">02 · MODEL RESULT</span><h2>{analysis?.classification?.stage || 'Awaiting analysis'}</h2></div><span className={`model-badge ${analysis?.classification?.active?'scenario':''}`}>{scenario?'SCENARIO':'LIVE / MODEL'}</span></div><div className="metric-row"><div><small>MODEL STATUS</small><b>{backend?'READY':'OFFLINE'}</b></div><div><small>OBSERVATION</small><b>{currentStatus.active_cyclone?'EVENT':'QUIET / NO CYCLONE'}</b></div><div><small>REGION</small><b>{region.name}</b></div></div><div className="forecast-toolbar"><div><span className="kicker">MODEL 2 · TEMPORAL FORECASTING</span><p>Historical replay from an unseen cyclone origin — track + intensity.</p></div><button className="primary" onClick={runForecast} disabled={forecastLoading}>{forecastLoading?<RefreshCw className="spin"/>:<Wind size={15}/>} {forecastLoading?'Forecasting…':'Run 6–48h forecast'}</button></div>{forecast?.error?<div className="forecast-error">{forecast.error}</div>:<div className="forecast-placeholder">{(forecast?.forecast||[{horizon_h:6},{horizon_h:12},{horizon_h:24},{horizon_h:48}]).map((f,i)=><div className="forecast-card" key={f.horizon_h}><span>+{String(f.horizon_h).padStart(2,'0')}H</span><b>{f.lat!=null?`${f.lat.toFixed(2)}°N · ${f.lon.toFixed(2)}°E`:'READY'}</b><small>{f.wind_kt!=null?`Wind ${f.wind_kt.toFixed(1)} kt · uncertainty ±${f.uncertainty_km.toFixed(0)} km`:'Click Run 6–48h forecast'}</small></div>)}</div>}{forecast?.origin_time&&<div className="forecast-origin"><span>HISTORICAL REPLAY</span><b>{forecast.sid}</b><small>Origin: {new Date(forecast.origin_time).toLocaleString()} · {forecast.model}</small></div>}</div>
    <div className="panel-xl glass"><div className="panel-head"><div><span className="kicker">03 · STATE IMPACT</span><h2>Dynamic state picture</h2></div><ShieldCheck size={18}/></div><div className="risk-list">{(analysis?.state_risk||REGIONS.map(r=>({state:r.state,zone:'GREEN'}))).slice(0,10).map((x,i)=><div className={`risk-line ${String(x.zone||'GREEN').toLowerCase()}`} key={i}><b>{x.state}</b><span>{x.zone}</span></div>)}</div><p className="fineprint">State zones remain neutral until a live/model event is actually returned. Scenario zones are not warnings.</p></div></section>
   <section className="panel-xl glass daily-panel"><div className="panel-head"><div><span className="kicker">04 · DAILY WEATHER OUTLOOK</span><h2>7-day local weather prediction</h2><p className="daily-subtitle">Short-range model outlook for {region.name}. This is probabilistic guidance, not a guarantee or an official IMD warning.</p></div><button className="primary" onClick={runDailyWeather} disabled={weatherLoading}>{weatherLoading?<RefreshCw className="spin"/>:<Globe2 size={15}/>} {weatherLoading?'Loading outlook…':'Refresh 7-day outlook'}</button></div>{dailyWeather?.error?<div className="forecast-error">{dailyWeather.error}</div>:<div className="daily-weather-grid">{(dailyWeather?.days||[]).map(d=><div className="daily-card" key={d.date}><span className="daily-date">{new Date(d.date+'T00:00:00').toLocaleDateString(undefined,{weekday:'short',day:'2-digit',month:'short'})}</span><b>{d.outlook}</b><strong>{d.temp_max_c}° / {d.temp_min_c}°C</strong><small>{d.condition}</small><div className="daily-stats"><span>🌧 {d.rain_probability}%</span><span>💧 {d.rain_mm} mm</span><span>💨 {d.wind_kmh} km/h</span></div></div>)}{!dailyWeather&&<div className="daily-empty"><span>7-DAY OUTLOOK</span><b>READY</b><small>Select a region and press “7-day weather” to load the latest model-based daily outlook.</small></div>}</div>}<div className="weather-source-note">Source: Open-Meteo weather models · Daily aggregation · <a href="https://open-meteo.com/en/docs" target="_blank" rel="noreferrer">methodology ↗</a> · Refresh when conditions change.</div></section>
   <section className="console-grid three-grid"><div className="panel-xl glass"><div className="panel-head"><span className="kicker">05 · FORECAST INTELLIGENCE</span><Globe2 size={18}/></div><div className="long-outlook">{(forecast?.forecast||[]).map(f=><div key={f.horizon_h}><span>+{f.horizon_h}H</span><b>{f.wind_kt?.toFixed(1) || '—'} KT</b><small>{f.lat?.toFixed(2)}°N · {f.lon?.toFixed(2)}°E · ±{f.uncertainty_km?.toFixed(0)} km</small></div>)}{!forecast&&<div><span>STATUS</span><b>READY</b><small>Run the forecast replay above to populate track, intensity and uncertainty.</small></div>}</div></div><div className="panel-xl glass"><div className="panel-head"><span className="kicker">06 · AI AGENT</span><Bot size={18}/></div><div className="agent-message">{agent || analysis?.ai_agent?.summary || 'Select a region and run analysis. The assistant will explain the current model result and readiness context.'}</div><div className="agent-actions"><button onClick={()=>setAgent('Check the latest IMD bulletin, satellite observation and official warning products before operational action.')}>What should officials check?</button><button onClick={()=>setAgent('If the observation changes materially, re-run the analysis. The risk layer should follow the new evidence rather than remain fixed.')}>What changes the result?</button><button onClick={()=>setAgent('Use the map to select Kochi, Visakhapatnam, Paradip, West Bengal Coast or an offshore Arabian Sea sector.')}>Show monitored regions</button></div></div><div className="panel-xl glass"><div className="panel-head"><span className="kicker">07 · REPORT</span><FileText size={18}/></div><h2>Decision summary</h2><p>{analysis?.ai_agent?.summary || 'No analysis has been run for this region yet.'}</p><ul className="summary-list">{(analysis?.ai_agent?.precautions||['Run an analysis first.','Verify official IMD guidance for operational decisions.']).map((x,i)=><li key={i}><CheckCircle2 size={14}/>{x}</li>)}</ul><button className="primary full" onClick={()=>window.print()}><FileText size={15}/> Print / Save PDF report</button></div></section>
   <section className="official-strip glass"><div><Radio size={17}/><b>Official-source layer</b><span>IMD Rapid Scan · INSAT-3DR imagery</span></div><a href={INSAT_SOURCE} target="_blank" rel="noreferrer">Open satellite service ↗</a><a href={IMD_SOURCE} target="_blank" rel="noreferrer">Open IMD ↗</a></section>
   <footer className="site-footer">CYCLONE-X · MODEL 1 CONNECTED · LIVE SATELLITE OBSERVATION SHOWN FROM OFFICIAL IMD SOURCE · NOT AN OFFICIAL WARNING SYSTEM</footer>
  </main>}
 </div>
}

createRoot(document.getElementById('root')).render(<App/>);
