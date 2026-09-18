import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, AlertTriangle, ArrowRight, Bell, Bot, Camera, CheckCircle2,
  ChevronRight, Clock3, Cpu, Gauge, LayoutDashboard, Map, Menu, MessageSquare,
  MonitorPlay, Radio, RefreshCw, ShieldAlert, SlidersHorizontal, Users, Wifi,
  X, Zap
} from 'lucide-react';
import './styles.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const WS = API.replace(/^http/, 'ws') + '/ws/live';

const nav = [
  ['Overview', LayoutDashboard], ['Live Monitoring', MonitorPlay], ['Customer Intelligence', Users],
  ['Customer Journey', Activity], ['Store Map', Map], ['Store Analytics', Gauge],
  ['Alerts', Bell], ['Camera Management', Camera], ['System Health', Cpu],
];

function useLiveData() {
  const [state, setState] = useState(null);
  const [online, setOnline] = useState(false);
  const [lastError, setLastError] = useState('');

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/v1/state`);
      if (!r.ok) throw new Error(`API ${r.status}`);
      setState(await r.json());
      setOnline(true); setLastError('');
    } catch (e) { setOnline(false); setLastError(e.message); }
  }, []);

  useEffect(() => {
    refresh();
    let socket;
    let retry;
    const connect = () => {
      try {
        socket = new WebSocket(WS);
        socket.onopen = () => setOnline(true);
        socket.onmessage = e => { try { setState(JSON.parse(e.data)); setOnline(true); } catch {} };
        socket.onerror = () => setOnline(false);
        socket.onclose = () => { setOnline(false); retry = setTimeout(connect, 2500); };
      } catch { setOnline(false); retry = setTimeout(connect, 2500); }
    };
    connect();
    return () => { clearTimeout(retry); socket?.close(); };
  }, [refresh]);
  return { state, online, lastError, refresh };
}

const fmtTime = ts => ts ? new Date(ts * 1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'}) : '—';
const fmtDuration = sec => { const s = Math.max(0, Math.round(sec || 0)); return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`; };

function Stat({ icon: Icon, label, value, detail, tone='' }) {
  return <div className="stat-card"><div className={`stat-icon ${tone}`}><Icon size={18}/></div><div className="stat-copy"><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></div></div>;
}
function Panel({title, subtitle, action, children, className=''}) {
  return <section className={`panel ${className}`}><div className="panel-head"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>;
}
function Empty({children='No data available yet.'}) { return <div className="empty">{children}</div>; }
function Badge({children, level=''}) { return <span className={`badge ${level}`}>{children}</span>; }

function Overview({data, online, go}) {
  const metrics = data?.metrics || {};
  const customers = data?.customers || [];
  const alerts = data?.alerts || [];
  const cameras = data?.cameras || [];
  const high = metrics.high_intent ?? customers.filter(x=>String(x.intent_level).toUpperCase()==='HIGH').length;
  return <>
    <section className="stats">
      <Stat icon={Camera} label="CAMERAS" value={`${cameras.filter(x=>x.status==='online').length}/${cameras.length || 1}`} detail={online ? 'Connected / configured' : 'Backend unavailable'} />
      <Stat icon={Users} label="ACTIVE CUSTOMERS" value={metrics.customers ?? customers.length} detail="Live tracked people" tone="blue" />
      <Stat icon={Zap} label="HIGH INTENT" value={high} detail="Requires attention" tone="amber" />
      <Stat icon={AlertTriangle} label="ACTIVE ALERTS" value={metrics.alerts ?? alerts.length} detail="Recent AI events" tone="red" />
    </section>
    <div className="grid-main">
      <Panel title="Camera Operations" subtitle="Live processing status" action={<button className="ghost-btn" onClick={()=>go('Live Monitoring')}>Open live <ArrowRight size={13}/></button>}>
        <div className="camera-grid">{cameras.length ? cameras.map(c=><div className="camera-card" key={c.camera_id}><div className="camera-preview"><Camera size={26}/><Badge level={c.status==='online'?'success':'danger'}>{c.status==='online'?'LIVE':'OFFLINE'}</Badge></div><div className="camera-meta"><b>{c.camera_id}</b><span>{c.fps || 0} FPS</span></div><small>{c.status==='online'?'AI processing':'Waiting for camera'}</small></div>) : <Empty>No cameras reported by the backend.</Empty>}</div>
      </Panel>
      <Panel title="Priority Alerts" subtitle="Events requiring attention" action={<button className="text-btn" onClick={()=>go('Alerts')}>View all</button>}>
        {alerts.slice(0,5).map(a=><AlertRow key={a.id || `${a.timestamp}-${a.customer_id}`} alert={a}/>)}{!alerts.length && <Empty>No active alerts.</Empty>}
      </Panel>
    </div>
    <div className="two-col"><Panel title="Live Intelligence" subtitle="Current AI state"><div className="mini-grid"><div><span>Camera</span><b>{data?.camera_id || '—'}</b></div><div><span>AI FPS</span><b>{data?.fps || 0}</b></div><div><span>Last update</span><b>{fmtTime(data?.last_update)}</b></div><div><span>Engine</span><b>YOLO + Tracking</b></div></div></Panel><Panel title="Intelligence Pipeline" subtitle="From camera signal to action"><div className="pipeline"><span>Camera</span><ArrowRight/><span>Detection</span><ArrowRight/><span>Tracking</span><ArrowRight/><span>Zones</span><ArrowRight/><span>Intent</span><ArrowRight/><span>Alerts</span></div></Panel></div>
  </>;
}
function AlertRow({alert:a}) { return <div className="alert-row"><div className={`alert-badge ${a.severity==='high'?'critical':''}`}><AlertTriangle size={13}/></div><div className="alert-body"><div><b>{a.type?.replaceAll('_',' ') || 'AI Alert'}</b><Badge level={a.severity==='high'?'danger':''}>{a.severity || 'info'}</Badge></div><p>{a.message || 'AI event detected.'}</p><small>{a.customer_id || a.camera_id || 'Camera'} · {fmtTime(a.timestamp)}</small></div></div>; }

function Live({data, online}) { const customers=data?.customers||[]; return <div className="live-layout"><Panel title="Live Camera" subtitle={`${data?.camera_id||'CAM_01'} · ${data?.fps||0} FPS`} className="video-panel"><div className="video-wrap">{online ? <img src={`${API}/api/v1/video.mjpg?ts=${Date.now()}`} alt="Live AI camera feed"/> : <div className="video-placeholder"><Radio size={36}/><b>Camera feed offline</b><span>Start the AI camera process to stream frames.</span></div>}<div className="video-overlay"><Badge level={online?'success':'danger'}>{online?'LIVE':'OFFLINE'}</Badge><span>{customers.length} tracked</span></div></div></Panel><Panel title="Live People" subtitle="Current tracked customers"><div className="people-list">{customers.map(c=><div className="person-row" key={c.customer_id}><div className="avatar"><Users size={15}/></div><div className="person-main"><b>{c.customer_id}</b><span>{c.zone} · {c.activity}</span></div><div className="person-side"><strong>{c.intent_score}</strong><small>{fmtDuration(c.dwell_seconds)}</small></div></div>)}{!customers.length&&<Empty>No tracked customers yet.</Empty>}</div></Panel></div>; }

function Customers({data}) { const rows=data?.customers||[]; return <Panel title="Customer Intelligence" subtitle="Live behavioral signals and intent"><div className="table-wrap"><table><thead><tr><th>Customer</th><th>Zone</th><th>Activity</th><th>Dwell</th><th>Intent</th><th>Signals</th></tr></thead><tbody>{rows.map(c=><tr key={c.customer_id}><td><b>{c.customer_id}</b></td><td>{c.zone}</td><td>{c.activity}</td><td>{fmtDuration(c.dwell_seconds)}</td><td><Badge level={String(c.intent_level).toUpperCase()==='HIGH'?'danger':String(c.intent_level).toUpperCase()==='MEDIUM'?'warning':''}>{c.intent_level} · {c.intent_score}</Badge></td><td>{[c.product_interaction&&'Product',c.phone_comparison&&'Phone',c.staff_nearby&&'Staff'].filter(Boolean).join(' · ')||'—'}</td></tr>)}</tbody></table>{!rows.length&&<Empty>No customers are being tracked.</Empty>}</div></Panel>; }

function Journey({data}) { const rows=data?.customers||[]; return <div className="journey-grid">{rows.map(c=><Panel key={c.customer_id} title={c.customer_id} subtitle={`${c.zone} · ${c.activity}`}><div className="journey"><div className="journey-step"><span className="step-dot"/><div><b>Current session</b><small>Live camera tracking</small></div></div><div className="journey-line"/><div className="journey-step"><span className="step-dot muted"/><div><b>{c.zone}</b><small>Dwell {fmtDuration(c.dwell_seconds)}</small></div></div><div className="journey-footer"><span>Intent</span><strong>{c.intent_score}/100</strong></div></div></Panel>)}{!rows.length&&<Panel title="Customer Journey" subtitle="Session paths will appear here"><Empty>Journey history will populate as customers are tracked.</Empty></Panel>}</div>; }

function StoreMap({data}) { const people=data?.customers||[]; return <div className="map-grid"><Panel title="Store Map" subtitle="Live customer and staff positions"><div className="store-map"><div className="zone z1"><b>Electronics</b>{people.filter((_,i)=>i%3===0).map(c=><span key={c.customer_id} className="map-person" title={c.customer_id}>●</span>)}</div><div className="zone z2"><b>Clothing</b>{people.filter((_,i)=>i%3===1).map(c=><span key={c.customer_id} className="map-person" title={c.customer_id}>●</span>)}</div><div className="zone z3"><b>Accessories</b>{people.filter((_,i)=>i%3===2).map(c=><span key={c.customer_id} className="map-person" title={c.customer_id}>●</span>)}</div><div className="zone z4"><b>Checkout</b></div><div className="entrance">ENTRANCE</div></div></Panel><Panel title="Zone Activity" subtitle="Current tracked density"><div className="bars">{['Electronics','Clothing','Accessories','Checkout'].map((z,i)=>{const n=people.filter(c=>String(c.zone).toLowerCase().includes(z.toLowerCase().slice(0,-1))).length; const width=Math.min(100, n*22+8); return <div className="bar-row" key={z}><span>{z}</span><div><i style={{width:`${width}%`}}/></div><b>{n}</b></div>})}</div></Panel></div>; }

function Analytics({data}) { const n=data?.customers?.length||0; const alerts=data?.alerts?.length||0; return <><div className="stats"><Stat icon={Users} label="LIVE FOOTFALL" value={n} detail="Current tracked customers"/><Stat icon={Clock3} label="AVG DWELL" value={n?fmtDuration((data.customers.reduce((a,c)=>a+(c.dwell_seconds||0),0)/n)):'0:00'} detail="Across active sessions"/><Stat icon={AlertTriangle} label="EVENTS" value={alerts} detail="Retained in live state"/><Stat icon={Gauge} label="PROCESSING FPS" value={data?.fps||0} detail="Current camera loop"/></div><div className="two-col"><Panel title="Customer Activity" subtitle="Live session distribution"><div className="activity-bars">{(data?.customers||[]).slice(0,8).map(c=><div key={c.customer_id}><span>{c.customer_id}</span><div><i style={{width:`${Math.min(100,Math.max(8,c.intent_score||8))}%`}}/></div><b>{c.intent_score}</b></div>)}{!n&&<Empty>Analytics will populate with live sessions.</Empty>}</div></Panel><Panel title="Operational Signals" subtitle="Current store intelligence"><div className="signal-cards"><div><Zap/><b>{data?.metrics?.high_intent||0}</b><span>High-intent customers</span></div><div><Bell/><b>{alerts}</b><span>Recent alerts</span></div><div><Users/><b>{n}</b><span>Active sessions</span></div></div></Panel></div></>; }

function Alerts({data}) { return <Panel title="Security & Retail Alerts" subtitle="Live AI events and operational notifications"><div className="alert-feed">{(data?.alerts||[]).map(a=><AlertRow key={a.id||a.timestamp} alert={a}/>)}{!(data?.alerts||[]).length&&<Empty>No alerts have been generated.</Empty>}</div></Panel>; }
function Cameras({data, refresh}) { const cams=data?.cameras||[]; return <Panel title="Camera Management" subtitle="Connected camera sources" action={<button className="ghost-btn" onClick={refresh}><RefreshCw size={13}/> Refresh</button>}><div className="camera-management">{cams.map(c=><div className="managed-camera" key={c.camera_id}><div className="camera-icon"><Camera/></div><div><b>{c.camera_id}</b><span>{c.status} · {c.fps||0} FPS</span></div><Badge level={c.status==='online'?'success':'danger'}>{c.status}</Badge><button className="icon-btn"><SlidersHorizontal size={15}/></button></div>)}{!cams.length&&<Empty>No camera configuration is available.</Empty>}</div></Panel>; }
function Health({data,online}) { const checks=[['FastAPI',online],['Live State',!!data],['Camera',data?.camera_status==='online'],['AI Processing',(data?.fps||0)>0],['WebSocket',online]]; return <><div className="health-grid">{checks.map(([name,ok])=><div className="health-card" key={name}><div className={`health-icon ${ok?'ok':'bad'}`}>{ok?<CheckCircle2/>:<AlertTriangle/>}</div><div><b>{name}</b><span>{ok?'Operational':'Unavailable'}</span></div></div>)}</div><Panel title="Runtime Diagnostics" subtitle="Current backend state"><div className="diagnostics"><div><span>Camera ID</span><b>{data?.camera_id||'—'}</b></div><div><span>Last update</span><b>{fmtTime(data?.last_update)}</b></div><div><span>FPS</span><b>{data?.fps||0}</b></div><div><span>Customers</span><b>{data?.metrics?.customers||0}</b></div></div></Panel></>; }

function App(){
  const {state,online,lastError,refresh}=useLiveData();
  const [page,setPage]=useState('Overview'); const [sidebar,setSidebar]=useState(false);
  const data=state||{metrics:{},customers:[],alerts:[],cameras:[],camera_status:'offline'};
  const content=useMemo(()=>({
    Overview:<Overview data={data} online={online} go={setPage}/>,
    'Live Monitoring':<Live data={data} online={online}/>,
    'Customer Intelligence':<Customers data={data}/>,
    'Customer Journey':<Journey data={data}/>,
    'Store Map':<StoreMap data={data}/>,
    'Store Analytics':<Analytics data={data}/>,
    Alerts:<Alerts data={data}/>,
    'Camera Management':<Cameras data={data} refresh={refresh}/>,
    'System Health':<Health data={data} online={online}/>,
  }[page]),[page,data,online,refresh]);
  return <div className="app-shell">
    <aside className={`sidebar ${sidebar?'open':''}`}><div className="brand"><div className="brand-mark">RA</div><div><b>Retail AI</b><span>Intelligence Platform</span></div></div><nav>{nav.map(([label,Icon])=><button key={label} className={page===label?'active':''} onClick={()=>{setPage(label);setSidebar(false)}}><Icon size={16}/><span>{label}</span>{label==='Alerts'&&data.alerts?.length>0?<em>{data.alerts.length}</em>:null}</button>)}</nav><div className="connection"><span className={`dot ${online?'online':''}`}/><span>API {online?'Connected':'Offline'}</span></div></aside>
    <main className="main"><header className="topbar"><button className="menu-btn" onClick={()=>setSidebar(!sidebar)}><Menu size={19}/></button><div><div className="eyebrow">STORE OPERATIONS · AI COMMAND CENTER</div><h1>{page}</h1></div><div className="top-actions"><button className="ghost-btn" onClick={refresh}><RefreshCw size={13}/> Sync</button><div className="live-pill"><span className={`dot ${online?'online':''}`}/>{online?'LIVE':'OFFLINE'}</div></div></header>{lastError&&!online&&<div className="offline-banner"><AlertTriangle size={15}/><span>Backend unavailable. Start the AI camera service on port 8000 to receive live data.</span></div>}<div className="page-content">{content}</div></main>
    {sidebar&&<div className="backdrop" onClick={()=>setSidebar(false)}/>}<button className="ai-assistant" title="Retail AI Assistant"><Bot size={19}/><span>AI</span></button>
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
