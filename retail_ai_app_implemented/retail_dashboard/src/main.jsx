import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, AlertTriangle, Bell, Bot, Camera, Check, ChevronRight, Clock3,
  Cpu, Gauge, LayoutDashboard, Map, Menu, MessageSquare, MonitorPlay, Radio,
  RefreshCw, ShieldAlert, Users, Wifi, X, Zap, UserRound, CircleAlert,
  Volume2, VolumeX, ArrowUpRight, UserCheck, UserX, Timer, BarChart3
} from 'lucide-react';
import './styles.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const WS = API.replace(/^http/, 'ws') + '/ws/live';
const VIDEO = `${API}/api/v1/video.mjpg`;

const NAV = [
  ['Overview', LayoutDashboard, 'COMMAND CENTER'],
  ['Live Floor', MonitorPlay, 'OPERATIONS'],
  ['Customer Intelligence', Users, 'OPERATIONS'],
  ['Customer Journey', Activity, 'OPERATIONS'],
  ['Store Map', Map, 'OPERATIONS'],
  ['Staff Operations', UserCheck, 'OPERATIONS'],
  ['Store Analytics', BarChart3, 'INTELLIGENCE'],
  ['AI Insights', Bot, 'INTELLIGENCE'],
  ['Security & Alerts', ShieldAlert, 'SECURITY'],
  ['Notifications', MessageSquare, 'SECURITY'],
  ['Camera Management', Camera, 'SYSTEM'],
  ['System Health', Cpu, 'SYSTEM'],
];

const fmtTime = (ts) => ts ? new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
const fmtDuration = (sec) => { const s = Math.max(0, Math.round(sec || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
const cap = (v) => String(v || '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());

function useLiveData() {
  const [data, setData] = useState(null);
  const [online, setOnline] = useState(false);
  const [error, setError] = useState('');
  const [lastPacket, setLastPacket] = useState(0);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/v1/state`, { cache: 'no-store' });
      if (!r.ok) throw new Error(`API ${r.status}`);
      const json = await r.json();
      setData(json); setOnline(true); setError(''); setLastPacket(Date.now());
    } catch (e) { setOnline(false); setError(e.message); }
  }, []);

  useEffect(() => {
    refresh();
    let socket; let retry;
    const connect = () => {
      try {
        socket = new WebSocket(WS);
        socket.onopen = () => { setOnline(true); setError(''); };
        socket.onmessage = (e) => {
          try { setData(JSON.parse(e.data)); setOnline(true); setLastPacket(Date.now()); } catch {}
        };
        socket.onerror = () => setOnline(false);
        socket.onclose = () => { setOnline(false); retry = setTimeout(connect, 2000); };
      } catch { setOnline(false); retry = setTimeout(connect, 2000); }
    };
    connect();
    return () => { clearTimeout(retry); socket?.close(); };
  }, [refresh]);

  return { data, online, error, lastPacket, refresh };
}

function Badge({ children, tone = '' }) { return <span className={`badge ${tone}`}>{children}</span>; }
function Stat({ icon: Icon, label, value, detail, tone = '' }) {
  return <div className="stat-card"><div className={`stat-icon ${tone}`}><Icon size={18} /></div><div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></div></div>;
}
function Panel({ title, subtitle, action, children, className = '' }) {
  return <section className={`panel ${className}`}><div className="panel-head"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>;
}
function Empty({ children = 'No live data available.' }) { return <div className="empty">{children}</div>; }

function AlertRow({ alert, onAck }) {
  const tone = alert.severity === 'critical' ? 'critical' : alert.severity === 'high' ? 'danger' : alert.severity === 'medium' ? 'warning' : '';
  return <div className={`alert-row ${alert.acknowledged ? 'acknowledged' : ''}`}>
    <div className={`alert-badge ${tone}`}><AlertTriangle size={13} /></div>
    <div className="alert-body">
      <div className="alert-title"><b>{cap(alert.type || 'AI Alert')}</b><Badge tone={tone}>{alert.severity || 'info'}</Badge></div>
      <p>{alert.message || 'AI event detected.'}</p>
      <small>{alert.zone || 'Store'} · {alert.camera_id || 'Camera'} · {fmtTime(alert.timestamp)}</small>
    </div>
    {!alert.acknowledged && alert.id && <button className="icon-btn" title="Acknowledge" onClick={() => onAck?.(alert.id)}><Check size={14} /></button>}
  </div>;
}

function Overview({ data, online, go, onAck }) {
  const m = data?.metrics || {}; const customers = data?.customers || []; const alerts = data?.alerts || [];
  const cameras = data?.cameras || []; const zones = data?.zone_stats || [];
  const high = m.high_intent ?? customers.filter(c => String(c.intent_level).toUpperCase() === 'HIGH').length;
  const staffConfigured = !!data?.staff_tracking_configured;
  const uncovered = staffConfigured ? (data?.staff_coverage || []).filter(z => z.customer_count > 0 && !z.staff_present).length : 0;
  return <>
    <div className="hero-strip"><div><div className="eyebrow">STORE COMMAND CENTER</div><h1>Live Retail Operations</h1><p>AI is continuously reading customer flow, zone activity and staff coverage.</p></div><div className="hero-actions"><Badge tone={online ? 'success' : 'danger'}><span className="live-dot" />{online ? 'AI LIVE' : 'API OFFLINE'}</Badge><button className="ghost-btn" onClick={() => go('Live Floor')}><MonitorPlay size={14}/> Open live floor</button></div></div>
    <section className="stats">
      <Stat icon={Users} label="LIVE VISITORS" value={m.customers ?? customers.length} detail="Currently tracked" tone="blue" />
      <Stat icon={Zap} label="HIGH INTENT" value={high} detail="Potential assistance" tone="amber" />
      <Stat icon={UserX} label="STAFF COVERAGE GAPS" value={uncovered} detail="Customer zones without staff" tone="red" />
      <Stat icon={AlertTriangle} label="LIVE ALERTS" value={m.alerts ?? alerts.filter(a => !a.acknowledged).length} detail="Requires attention" tone="red" />
    </section>
    <div className="command-grid">
      <Panel title="Live Floor" subtitle={`${data?.camera_id || 'CAM_01'} · ${data?.fps || 0} FPS`} className="command-video" action={<button className="text-btn" onClick={() => go('Live Floor')}>Full screen <ArrowUpRight size={13}/></button>}>
        <div className="video-mini"><img src={`${VIDEO}?overview=1`} alt="Live AI camera" /><div className="video-badge"><span className="live-dot"/> LIVE</div><div className="track-count">{customers.length} tracked</div></div>
      </Panel>
      <Panel title="Live Alerts" subtitle="AI events requiring action" action={<button className="text-btn" onClick={() => go('Security & Alerts')}>View all</button>}>
        <div className="alert-feed">{alerts.slice(0, 5).map(a => <AlertRow key={a.id} alert={a} onAck={onAck}/>)}{!alerts.length && <Empty>No active alerts. The store is clear.</Empty>}</div>
      </Panel>
    </div>
    <div className="three-col">
      <Panel title="Zone Coverage" subtitle={data?.staff_tracking_configured ? 'Customer presence vs registered staff presence' : 'Register staff tracker IDs to enable absence alerts'}><ZoneBars zones={zones} staffConfigured={!!data?.staff_tracking_configured}/></Panel>
      <Panel title="Store Pulse" subtitle="Current operating signals"><div className="pulse-grid"><div><Users/><b>{customers.length}</b><span>Visitors</span></div><div><Clock3/><b>{fmtDuration(customers.reduce((s,c)=>s+(c.dwell_seconds||0),0)/Math.max(customers.length,1))}</b><span>Avg dwell</span></div><div><Gauge/><b>{data?.fps || 0}</b><span>AI FPS</span></div><div><ShieldAlert/><b>{alerts.length}</b><span>Events</span></div></div></Panel>
      <Panel title="AI Pipeline" subtitle="Signal → decision → action"><div className="pipeline vertical"><span>Camera stream</span><ChevronRight/><span>Detection + tracking</span><ChevronRight/><span>Zone + dwell</span><ChevronRight/><span>Intent + staff coverage</span><ChevronRight/><span>Live alert</span></div></Panel>
    </div>
  </>;
}

function ZoneBars({ zones = [], staffConfigured = true }) {
  if (!zones.length) return <Empty>Zone statistics appear when the camera is running.</Empty>;
  return <div className="zone-bars">{zones.map(z => { const intensity = Math.min(100, (z.customer_count || 0) * 18); return <div className="zone-row" key={z.zone}><div className="zone-label"><b>{z.zone}</b><span>{z.customer_count || 0} customers · {z.staff_present ? 'staff present' : 'staff absent'}</span></div><div className="zone-track"><i style={{ width: `${Math.max(4, intensity)}%` }}/></div><Badge tone={!staffConfigured ? '' : z.staff_present ? 'success' : z.customer_count ? 'danger' : ''}>{!staffConfigured ? 'Unconfigured' : z.staff_present ? 'Covered' : z.customer_count ? 'Gap' : 'Clear'}</Badge></div>; })}</div>;
}

function LiveFloor({ data, online, onAck }) {
  const [sound, setSound] = useState(false); const lastAlert = useRef('');
  useEffect(() => {
    const a = data?.alerts?.[0];
    if (sound && a && a.id !== lastAlert.current && (a.severity === 'critical' || a.severity === 'high')) {
      try { const ctx = new AudioContext(); const osc = ctx.createOscillator(); const gain = ctx.createGain(); osc.frequency.value = 720; gain.gain.value = 0.035; osc.connect(gain); gain.connect(ctx.destination); osc.start(); osc.stop(ctx.currentTime + 0.18); } catch {}
    }
    if (a) lastAlert.current = a.id;
  }, [data?.alerts, sound]);
  const customers = data?.customers || []; const coverage = data?.staff_coverage || [];
  return <div className="live-layout">
    <Panel title="Live Camera Intelligence" subtitle={`${data?.camera_id || 'CAM_01'} · ${data?.fps || 0} FPS`} action={<button className="ghost-btn" onClick={() => setSound(v => !v)}>{sound ? <Volume2 size={14}/> : <VolumeX size={14}/>} {sound ? 'Alert sound on' : 'Alert sound off'}</button>}>
      <div className="video-wrap large"><img src={`${VIDEO}?live=1`} alt="Live AI camera feed"/><div className="video-overlay"><Badge tone={online ? 'success' : 'danger'}>{online ? 'LIVE' : 'OFFLINE'}</Badge><span>{customers.length} tracked</span><span>{data?.fps || 0} FPS</span></div></div>
    </Panel>
    <div className="live-side">
      <Panel title="Immediate Attention" subtitle="Newest operational events"><div className="alert-feed">{(data?.alerts || []).slice(0, 7).map(a => <AlertRow key={a.id} alert={a} onAck={onAck}/>)}{!(data?.alerts || []).length && <Empty>Waiting for events.</Empty>}</div></Panel>
      <Panel title="Zone Coverage" subtitle="Staff presence monitoring"><ZoneBars zones={coverage}/></Panel>
    </div>
    <Panel title="Tracked People" subtitle="Current camera tracks" className="full-span"><div className="people-grid">{customers.map(c => <div className="person-card" key={c.customer_id}><div className="person-head"><div className="avatar"><UserRound size={15}/></div><div><b>{c.customer_id}</b><span>{c.zone}</span></div><Badge tone={String(c.intent_level).toUpperCase()==='HIGH'?'danger':String(c.intent_level).toUpperCase()==='MEDIUM'?'warning':''}>{c.intent_level}</Badge></div><div className="person-metrics"><div><span>Activity</span><b>{c.activity}</b></div><div><span>Dwell</span><b>{fmtDuration(c.dwell_seconds)}</b></div><div><span>Intent</span><b>{c.intent_score}/100</b></div><div><span>Staff</span><b className={c.staff_nearby === false ? 'text-danger' : ''}>{c.staff_nearby === false ? 'Not nearby' : c.staff_nearby === true ? 'Nearby' : 'Unknown'}</b></div></div></div>)}{!customers.length && <Empty>No people detected yet.</Empty>}</div></Panel>
  </div>;
}

function Customers({ data }) { const rows = data?.customers || []; return <Panel title="Customer Intelligence" subtitle="Live behavioral signals and assistance opportunities"><div className="table-wrap"><table><thead><tr><th>Customer</th><th>Zone</th><th>Activity</th><th>Dwell</th><th>Intent</th><th>Signals</th><th>Staff</th></tr></thead><tbody>{rows.map(c => <tr key={c.customer_id}><td><b>{c.customer_id}</b></td><td>{c.zone}</td><td>{c.activity}</td><td>{fmtDuration(c.dwell_seconds)}</td><td><Badge tone={String(c.intent_level).toUpperCase()==='HIGH'?'danger':String(c.intent_level).toUpperCase()==='MEDIUM'?'warning':''}>{c.intent_level} · {c.intent_score}</Badge></td><td>{[c.product_interaction&&'Product',c.phone_comparison&&'Phone'].filter(Boolean).join(' · ') || 'Browsing'}</td><td><span className={c.staff_nearby===false?'text-danger':''}>{c.staff_nearby===false?'Absent':c.staff_nearby===true?'Nearby':'Unknown'}</span></td></tr>)}</tbody></table>{!rows.length&&<Empty>No customers are being tracked.</Empty>}</div></Panel>; }

function Journey({ data }) { const rows = data?.customers || []; return <div className="journey-grid">{rows.map(c => <Panel key={c.customer_id} title={c.customer_id} subtitle={`${c.zone} · ${c.activity}`}><div className="journey"><div className="journey-step"><span className="step-dot"/><div><b>Live session</b><small>Camera {c.camera_id}</small></div></div><div className="journey-line"/><div className="journey-step"><span className="step-dot"/><div><b>{c.zone}</b><small>Current zone · dwell {fmtDuration(c.dwell_seconds)}</small></div></div><div className="journey-line"/><div className="journey-step"><span className="step-dot muted"/><div><b>{c.staff_nearby === false ? 'Assistance gap' : 'Service available'}</b><small>{c.staff_nearby === false ? 'No nearby staff detected' : 'Staff signal present or unknown'}</small></div></div><div className="journey-footer"><span>Intent score</span><strong>{c.intent_score}/100</strong></div></div></Panel>)}{!rows.length&&<Panel title="Customer Journey" subtitle="Live paths and service moments"><Empty>Journey data will populate as customers move through zones.</Empty></Panel>}</div>; }

function StoreMap({ data }) { const zones = data?.zone_stats || []; const people = data?.customers || []; return <div className="map-grid"><Panel title="Store Map" subtitle="Live zone occupancy and coverage"><div className="store-map">{zones.map((z, i) => <div className={`floor-zone zone-${i % 6}`} key={z.zone}><div className="floor-zone-title"><b>{z.zone}</b><span>{z.customer_count || 0} visitors</span></div><div className="map-dots">{people.filter(p => p.zone === z.zone).map(p => <span key={p.customer_id} title={`${p.customer_id} · ${p.activity}`} className={`map-dot ${String(p.intent_level).toLowerCase()}`}/>)}</div><div className="zone-footer"><span>{z.staff_present ? 'Staff present' : z.customer_count ? 'STAFF GAP' : 'No traffic'}</span><Badge tone={z.staff_present ? 'success' : z.customer_count ? 'danger' : ''}>{z.staff_present ? 'Covered' : z.customer_count ? 'Alert' : 'Clear'}</Badge></div></div>)}{!zones.length && <Empty>Run the camera to populate the store map.</Empty>}</div></Panel><Panel title="Zone Detail" subtitle="Current operational state"><ZoneBars zones={zones}/></Panel></div>; }

function StaffOps({ data }) { const zones = data?.staff_coverage || []; const configured = !!data?.staff_tracking_configured; const gaps = configured ? zones.filter(z => z.customer_count > 0 && !z.staff_present) : []; return <><section className="stats"><Stat icon={UserCheck} label="STAFF PRESENT" value={zones.filter(z=>z.staff_present).length} detail="Covered zones" tone="blue"/><Stat icon={UserX} label="COVERAGE GAPS" value={gaps.length} detail="Customer zones without staff" tone="red"/><Stat icon={Users} label="VISITORS IN GAPS" value={gaps.reduce((s,z)=>s+z.customer_count,0)} detail="Need attention" tone="amber"/><Stat icon={Timer} label="ALERT MODE" value="LIVE" detail="Camera-driven monitoring" tone="green"/></section><div className="two-col"><Panel title="Staff Coverage by Zone" subtitle={configured ? 'A zone is flagged when customers are present and no registered staff tracker is detected.' : 'Staff tracker IDs are not configured yet.'}><ZoneBars zones={zones} staffConfigured={configured}/></Panel><Panel title="Coverage Gaps" subtitle="Operational alerts for the floor team">{gaps.length ? gaps.map(g => <div className="gap-card" key={g.zone}><div className="gap-icon"><UserX size={16}/></div><div><b>{g.zone}</b><span>{g.customer_count} customer{g.customer_count===1?'':'s'} currently present</span></div><Badge tone="danger">Staff absent</Badge></div>) : <Empty>No current staff coverage gaps.</Empty>}</Panel></div></>; }

function Analytics({ data }) { const zones = data?.zone_stats || []; const customers = data?.customers || []; const max = Math.max(1, ...zones.map(z=>z.customer_count||0)); return <><div className="three-col"><Panel title="Footfall Snapshot" subtitle="Live tracked visitors"><div className="big-number">{customers.length}</div><div className="muted">Active people in the camera field</div></Panel><Panel title="Average Dwell" subtitle="Across active customers"><div className="big-number">{fmtDuration(customers.reduce((s,c)=>s+(c.dwell_seconds||0),0)/Math.max(1,customers.length))}</div><div className="muted">Current session average</div></Panel><Panel title="High Intent" subtitle="Customers with high intent signal"><div className="big-number">{customers.filter(c=>String(c.intent_level).toUpperCase()==='HIGH').length}</div><div className="muted">Potential service opportunities</div></Panel></div><Panel title="Zone Activity" subtitle="Live occupancy distribution"><div className="analytics-bars">{zones.map(z=><div className="analytics-row" key={z.zone}><div><b>{z.zone}</b><span>{z.customer_count} visitors</span></div><div className="analytics-track"><i style={{width:`${Math.max(3,(z.customer_count/max)*100)}%`}}/></div><strong>{z.customer_count}</strong></div>)}{!zones.length&&<Empty>No zone analytics yet.</Empty>}</div></Panel></>; }

function Insights({ data, go }) { const zones = data?.zone_stats || []; const configured=!!data?.staff_tracking_configured; const gaps = configured ? (data?.staff_coverage || []).filter(z=>z.customer_count>0&&!z.staff_present) : []; const high = (data?.customers||[]).filter(c=>String(c.intent_level).toUpperCase()==='HIGH'); return <div className="insight-grid"><Panel title="AI Operations Brief" subtitle="Generated from the current live state"><div className="insight-list">{gaps.length ? gaps.map(g=><div className="insight-card danger" key={`gap-${g.zone}`}><div className="insight-icon"><UserX size={17}/></div><div><b>Staff coverage gap in {g.zone}</b><p>{g.customer_count} customer{g.customer_count===1?' is':'s are'} present without a registered staff tracker. Route an associate to the zone.</p><button className="text-btn" onClick={()=>go('Staff Operations')}>Open staff operations <ArrowUpRight size={12}/></button></div></div>) : <div className="insight-card success"><div className="insight-icon"><Check size={17}/></div><div><b>No current coverage gaps</b><p>Every zone with active customer traffic has a registered staff presence signal.</p></div></div>}{high.length ? <div className="insight-card amber"><div className="insight-icon"><Zap size={17}/></div><div><b>{high.length} high-intent customer{high.length===1?'':'s'} detected</b><p>Review their zones and staff proximity for service opportunities.</p><button className="text-btn" onClick={()=>go('Customer Intelligence')}>Review customers <ArrowUpRight size={12}/></button></div></div> : null}{zones.length ? <div className="insight-card"><div className="insight-icon"><Map size={17}/></div><div><b>Zone activity is being monitored</b><p>{zones.filter(z=>z.customer_count>0).length} zones currently have customer traffic.</p></div></div> : null}</div></Panel><Panel title="AI Assistant" subtitle="Ask about the current store state"><div className="assistant-box"><Bot size={22}/><div><b>Retail floor assistant</b><p>Try questions like “Which zone needs staff?” or “Who needs assistance?”</p><div className="suggestions"><button>Which zone needs staff?</button><button>Who needs assistance?</button><button>Show current alerts</button></div></div></div></Panel></div>; }

function Notifications({ data }) {
  const n = data?.notification_status || {};
  const gaps = (data?.staff_coverage || []).filter(z => z.customer_count > 0 && !z.staff_present);
  return <div className="notification-page">
    <section className="stats">
      <Stat icon={MessageSquare} label="WHATSAPP" value={n.configured ? 'READY' : n.enabled ? 'SETUP' : 'OFF'} detail={n.configured ? 'Twilio connected' : 'Optional staff alert channel'} tone={n.configured ? 'green' : 'amber'} />
      <Stat icon={Bell} label="RECIPIENTS" value={n.recipient_count || 0} detail="Configured broadcast recipients" tone="blue" />
      <Stat icon={Zap} label="MESSAGES SENT" value={n.sent_count || 0} detail="This process session" tone="green" />
      <Stat icon={AlertTriangle} label="CURRENT GAPS" value={gaps.length} detail="Zones requiring coverage" tone="red" />
    </section>
    <div className="two-col">
      <Panel title="WhatsApp Staff Alerts" subtitle="Automatic escalation when customer zones have no registered staff">
        <div className="notification-card">
          <div className={`notification-status ${n.configured ? 'ok' : 'warn'}`}><MessageSquare size={18}/><div><b>{n.configured ? 'WhatsApp alerting is connected' : 'WhatsApp alerting is not connected'}</b><span>{n.configured ? `${n.recipient_count || 0} default recipient(s) · ${n.zone_recipient_groups || 0} zone routing group(s)` : 'The AI dashboard will still show alerts without WhatsApp.'}</span></div></div>
          <div className="notification-rules">
            <div><b>1 · Coverage gap</b><span>Customers detected + no registered staff → alert after confirmation window.</span></div>
            <div><b>2 · Escalation</b><span>Persistent gap → critical escalation after the configured threshold.</span></div>
            <div><b>3 · Queue pressure</b><span>Billing queue above threshold → operational notification.</span></div>
          </div>
        </div>
      </Panel>
      <Panel title="Current Coverage Gaps" subtitle="These zones are candidates for staff notification">
        {gaps.length ? gaps.map(g => <div className="gap-card" key={g.zone}><div className="gap-icon"><UserX size={16}/></div><div><b>{g.zone}</b><span>{g.customer_count} customer{g.customer_count === 1 ? '' : 's'} · gap {fmtDuration(g.gap_seconds)}</span></div><Badge tone="danger">Notify</Badge></div>) : <Empty>No active coverage gaps.</Empty>}
      </Panel>
    </div>
    {n.last_error && <Panel title="Notification Diagnostics" subtitle="Most recent delivery error"><div className="empty">{n.last_error}</div></Panel>}
  </div>; }


function Alerts({ data, onAck }) { const alerts=data?.alerts||[]; return <Panel title="Security & Live Alerts" subtitle="Real-time events from the AI pipeline"><div className="alert-feed large-feed">{alerts.map(a=><AlertRow key={a.id} alert={a} onAck={onAck}/>)}{!alerts.length&&<Empty>No alerts at the moment.</Empty>}</div></Panel>; }

function Cameras({ data, online, go }) { const cams=data?.cameras||[]; return <div className="camera-management"><Panel title="Camera Management" subtitle="Connected AI camera sources"><div className="camera-list">{cams.map(c=><div className="managed-camera" key={c.camera_id}><div className="camera-icon"><Camera size={17}/></div><div><b>{c.camera_id}</b><span>{c.status} · {c.fps||0} FPS</span></div><Badge tone={c.status==='online'?'success':'danger'}>{c.status}</Badge><button className="ghost-btn" onClick={()=>go('Live Floor')}>View</button></div>)}{!cams.length&&<Empty>{online?'No camera configuration reported.':'Connect the backend to discover cameras.'}</Empty>}</div></Panel></div>; }

function Health({ data, online, lastPacket }) { const checks=[['FastAPI',online],['WebSocket',online],['Camera',data?.camera_status==='online'],['AI stream',(data?.fps||0)>0],['Live state',Date.now()-lastPacket<3000]]; return <><div className="health-grid">{checks.map(([name,ok])=><div className="health-card" key={name}><div className={`health-icon ${ok?'ok':'bad'}`}>{ok?<Check size={15}/>:<X size={15}/>}</div><div><b>{name}</b><span>{ok?'Healthy':'Waiting'}</span></div></div>)}</div><div className="two-col"><Panel title="Runtime" subtitle="Current engine telemetry"><div className="diagnostics"><div><span>Camera</span><b>{data?.camera_id||'—'}</b></div><div><span>AI FPS</span><b>{data?.fps||0}</b></div><div><span>Last update</span><b>{fmtTime(data?.last_update)}</b></div><div><span>Tracked</span><b>{data?.customers?.length||0}</b></div></div></Panel><Panel title="Alerting" subtitle="Operational notification state"><div className="diagnostics"><div><span>Active events</span><b>{data?.alerts?.length||0}</b></div><div><span>Coverage gaps</span><b>{data?.staff_tracking_configured ? (data?.staff_coverage||[]).filter(z=>z.customer_count>0&&!z.staff_present).length : '—'}</b></div><div><span>Event buffer</span><b>{data?.events?.length||0}</b></div><div><span>Transport</span><b>{online?'WebSocket':'Offline'}</b></div></div></Panel></div></>; }

function App() {
  const { data, online, error, lastPacket, refresh } = useLiveData();
  const [page, setPage] = useState('Overview'); const [mobile, setMobile] = useState(false); const [toast, setToast] = useState('');
  const ack = async (id) => { try { const r=await fetch(`${API}/api/v1/alerts/${encodeURIComponent(id)}/ack`, {method:'POST'}); if(!r.ok) throw new Error(); setToast('Alert acknowledged'); setTimeout(()=>setToast(''),1800); } catch { setToast('Could not acknowledge alert'); setTimeout(()=>setToast(''),1800); } };
  const go = (p) => { setPage(p); setMobile(false); };
  const content = useMemo(() => {
    const common={data,online,go,onAck:ack};
    switch(page){
      case 'Live Floor': return <LiveFloor {...common}/>;
      case 'Customer Intelligence': return <Customers {...common}/>;
      case 'Customer Journey': return <Journey {...common}/>;
      case 'Store Map': return <StoreMap {...common}/>;
      case 'Staff Operations': return <StaffOps {...common}/>;
      case 'Store Analytics': return <Analytics {...common}/>;
      case 'AI Insights': return <Insights {...common}/>;
      case 'Security & Alerts': return <Alerts {...common}/>;
      case 'Notifications': return <Notifications {...common}/>;
      case 'Camera Management': return <Cameras {...common}/>;
      case 'System Health': return <Health {...common} lastPacket={lastPacket}/>;
      default: return <Overview {...common}/>;
    }
  }, [page,data,online,lastPacket]);
  const group = {}; NAV.forEach(n=>(group[n[2]]??=[]).push(n));
  return <div className="app-shell">
    {mobile&&<div className="backdrop" onClick={()=>setMobile(false)}/>}<aside className={`sidebar ${mobile?'open':''}`}>
      <div className="brand"><div className="brand-mark">RA</div><div><b>Retail AI</b><span>Store Intelligence</span></div></div>
      <div className="nav-scroll">{Object.entries(group).map(([label,items])=><div className="nav-group" key={label}><div className="nav-label">{label}</div>{items.map(([name,Icon])=><button key={name} className={`nav-item ${page===name?'active':''}`} onClick={()=>go(name)}><Icon size={15}/><span>{name}</span>{name==='Security & Alerts'&&data?.alerts?.length?<em>{data.alerts.length}</em>:null}</button>)}</div>)}</div>
      <div className="connection"><span className={`dot ${online?'online':''}`}/>{online?'AI API Connected':'AI API Offline'}<button className="refresh-mini" onClick={refresh}><RefreshCw size={12}/></button></div>
    </aside>
    <main className="main"><header className="topbar"><button className="menu-btn" onClick={()=>setMobile(true)}><Menu size={18}/></button><div className="crumb"><span>Retail AI</span><ChevronRight size={13}/><b>{page}</b></div><div className="top-actions"><div className="sync"><span className={`dot ${online?'online':''}`}/>{online?'LIVE':'OFFLINE'}{error&&<small>{error}</small>}</div><button className="ghost-btn" onClick={refresh}><RefreshCw size={13}/> Sync</button></div></header><div className="page-content">{content}</div></main>
    {toast&&<div className="toast"><Check size={15}/>{toast}</div>}
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
