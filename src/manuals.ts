import manifest from "../data/manuals.json";

export type Manual = {
  id: string;
  label: string;
  title: string;
  filename: string;
  jurisdiction: string;
  url: string;
};

export const manuals: Manual[] = manifest.map((manual) => ({
  ...manual,
  url: `/api/manuals/${manual.id}/pdf`,
}));

export const manualIds: ReadonlySet<string> = new Set(
  manuals.map((manual) => manual.id),
);
