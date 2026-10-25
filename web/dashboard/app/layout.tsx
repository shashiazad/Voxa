import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "Voxa Console", description: "Voice support agents for your organization" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
