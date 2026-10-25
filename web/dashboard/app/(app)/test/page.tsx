import CallPanel from "@/components/CallPanel";

export default function TestCall() {
  return (<>
    <div className="page-head"><div><h1>Test call</h1><p className="muted" style={{ margin: 0 }}>Talk to your agent in the browser and watch latency live.</p></div></div>
    <CallPanel />
  </>);
}
