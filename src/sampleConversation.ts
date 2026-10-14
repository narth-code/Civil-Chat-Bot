/** Static, labelled demonstration conversation shown outside live history. */
import { type ManualId } from "./api";

export const sampleId = "sample-workflow";

export type SampleMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  selected_manual_ids: ManualId[];
  sample: true;
};

export const sample = {
  id: sampleId,
  title: "Sample: review workflow",
  messages: [
    {
      id: "sample-q",
      role: "user" as const,
      content:
        "How should I use this workspace when reviewing a transportation submittal?",
      selected_manual_ids: [] as ManualId[],
      sample: true as const,
    },
    {
      id: "sample-a",
      role: "assistant" as const,
      content:
        "Start by choosing the applicable manual at left, then use a focused question to identify passages for review. Treat returned sources as a reading queue: open the original PDF, confirm the governing edition and context, and document your own project-specific determination. This demonstration does not verify requirements, page numbers, or section references.",
      selected_manual_ids: ["transportation"] as ManualId[],
      sample: true as const,
    },
  ],
};
