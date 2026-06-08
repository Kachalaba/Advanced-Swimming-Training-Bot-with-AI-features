"use client";

import { useParams } from "next/navigation";

import { SwimmingAnalysisWorkspace } from "@/components/swimming/SwimmingAnalysisWorkspace";

export default function SwimmingResultPage() {
  const params = useParams<{ jobId: string }>();
  return <SwimmingAnalysisWorkspace jobId={params.jobId} />;
}
