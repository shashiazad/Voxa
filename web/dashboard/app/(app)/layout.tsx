import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import Shell from "@/components/Shell";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  if (!(await cookies()).get("vx_session")) redirect("/login");
  return <Shell>{children}</Shell>;
}
