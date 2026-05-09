"use client";

import { useEffect, useState } from "react";

import { DashboardView } from "@/components/dashboard/DashboardView";
import { api } from "@/lib/api";

export default function HomePage() {
  const [name, setName] = useState("Athlete");
  useEffect(() => {
    api
      .me()
      .then((m) => setName(m.name))
      .catch(() => undefined);
  }, []);

  return <DashboardView athleteName={name} />;
}
