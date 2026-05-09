"use client";

import { useEffect, useState } from "react";

import { DashboardView } from "@/components/dashboard/DashboardView";
import { TopNav } from "@/components/layout/TopNav";
import { api } from "@/lib/api";

export default function HomePage() {
  const [name, setName] = useState("Athlete");
  useEffect(() => {
    api
      .me()
      .then((m) => setName(m.name))
      .catch(() => undefined);
  }, []);

  return (
    <>
      <TopNav />
      <main className="max-w-[1600px] mx-auto px-6 py-8">
        <DashboardView athleteName={name} />
      </main>
    </>
  );
}
