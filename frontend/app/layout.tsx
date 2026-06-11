import type { Metadata } from "next";
import { Inter } from "next/font/google";

import { TopNav } from "@/components/layout/TopNav";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "SPRINT AI · Triathlon Intelligence",
  description:
    "AI-powered video analysis for swimming, running, cycling, and dryland training.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`dark ${inter.variable}`}>
      <body className="bg-bg text-slate-100 font-sans">
        <TopNav />
        <main className="mx-auto max-w-[1600px] px-3 py-4 sm:px-6 sm:py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
