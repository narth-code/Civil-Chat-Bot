export type ManualId = "transportation" | "environmental" | "fire";

export type DemoDay = {
  date: string;
  questions: number;
  engineerIds: string[];
  responseSamples: number[];
  unanswered: number;
  retrievalRequests: number;
  retrievalLatencies: number[];
  sourceReturns: number;
  noMatches: number;
  manualPassages: Record<ManualId, number>;
  manualReferences: Record<ManualId, number>;
  unansweredReasons: Record<
    "No matching passage" | "Needs project context" | "Out of scope",
    number
  >;
};

const manuals: ManualId[] = ["transportation", "environmental", "fire"];
const reasons = [
  "No matching passage",
  "Needs project context",
  "Out of scope",
] as const;

/** Fixed, deterministic demonstration activity for Apr 2–Jun 30, 2026. */
export const demoActivity: DemoDay[] = Array.from(
  { length: 90 },
  (_, index) => {
    const date = new Date(Date.UTC(2026, 3, 2 + index));
    const questions = 17 + ((index * 11 + (index % 6) * 3) % 22);
    const unanswered = 1 + ((index * 5 + 2) % 5);
    const retrievalRequests = questions + 5 + (index % 8);
    const noMatches = (index * 7 + 1) % 4;
    const sourceReturns = retrievalRequests - noMatches;
    const latencies = Array.from(
      { length: retrievalRequests },
      (_, sample) =>
        185 + ((index * 37 + sample * 29 + (sample % 4) * 41) % 590),
    );
    const passageTotal = retrievalRequests * 2 + ((index * 3) % 9);
    const manualPassages = {
      transportation: Math.floor(passageTotal * (0.47 + (index % 3) * 0.02)),
      environmental: Math.floor(
        passageTotal * (0.31 + ((index + 1) % 3) * 0.015),
      ),
      fire: 0,
    };
    manualPassages.fire =
      passageTotal -
      manualPassages.transportation -
      manualPassages.environmental;
    const manualReferences = {
      transportation: Math.max(1, Math.round(questions * 0.52) + (index % 4)),
      environmental: Math.max(
        1,
        Math.round(questions * 0.29) + ((index + 2) % 3),
      ),
      fire: Math.max(1, Math.round(questions * 0.17) + (index % 2)),
    };
    const unansweredReasons = {
      "No matching passage": Math.max(
        0,
        Math.min(unanswered, (index * 3 + 1) % 4),
      ),
      "Needs project context": 0,
      "Out of scope": 0,
    };
    unansweredReasons["Needs project context"] = Math.max(
      0,
      unanswered - unansweredReasons["No matching passage"] - (index % 2),
    );
    unansweredReasons["Out of scope"] =
      unanswered -
      unansweredReasons["No matching passage"] -
      unansweredReasons["Needs project context"];
    return {
      date: date.toISOString().slice(0, 10),
      questions,
      engineerIds: Array.from(
        { length: 8 + ((index * 3 + 4) % 12) },
        (_, engineer) =>
          `ENG-${String(((index * 5 + engineer * 3) % 24) + 1).padStart(2, "0")}`,
      ),
      responseSamples: Array.from(
        { length: questions - unanswered },
        (_, sample) => 640 + ((index * 53 + sample * 31) % 1100),
      ),
      unanswered,
      retrievalRequests,
      retrievalLatencies: latencies,
      sourceReturns,
      noMatches,
      manualPassages,
      manualReferences,
      unansweredReasons,
    };
  },
);

export const demoPeriodLabel = "Apr 2 – Jun 30, 2026";
export const manualIds = manuals;
export const unansweredReasonLabels = reasons;
