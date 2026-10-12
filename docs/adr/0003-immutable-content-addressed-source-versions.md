# Source versions are immutable, content-addressed copies

Each Source version points at a copy of the PDF stored under its SHA-256 hash, not at the original file in `docs/`. A citation must keep pointing at the exact bytes it was made from, and an original file can be replaced or renamed without anyone noticing. Registering the same bytes again reuses the existing version; changed bytes become a new Source version.

A Source version is identified by the date it took effect (for example a manual verified as the July 2025 edition). When a newer effective edition is published, it is registered as a new Source version and re-parsed; the old one is kept. The date the file was uploaded is recorded separately as the Capture date, to track data drift. Edition, effective dates, and applicability notes are stored as unknown until a person records them. They are never inferred from a filename.
