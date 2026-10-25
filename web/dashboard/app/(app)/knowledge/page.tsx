"use client";
import KnowledgeManager from "@/components/KnowledgeManager";
import { canEdit, useMe } from "@/lib/hooks";

export default function Knowledge() {
  const me = useMe();
  return (<>
    <div className="page-head"><div><h1>Knowledge base</h1>
      <p className="muted" style={{ margin: 0 }}>The agent answers only from this content and its tools. Changes apply to the next call.</p></div></div>
    <KnowledgeManager readOnly={!!me && !canEdit(me.role)} />
  </>);
}
