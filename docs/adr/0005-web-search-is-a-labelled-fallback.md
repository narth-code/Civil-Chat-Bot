# Web search is a labelled fallback, not authoritative evidence

When the database has no sufficient Evidence, the assistant searches the web, restricted to official government sites, and returns the result as Web evidence: labelled as coming from web search, with the page URL as the evidence path. Dropping web search from the MVP was proposed and rejected by the product owner.

Consequences: a search query leaves the machine, which narrows ADR-0001 for that one step. Web evidence belongs to no registered Source version, so it has no edition or effective date and cannot be checked for Applicability. It is never presented as if it were a quote from a Source version.
