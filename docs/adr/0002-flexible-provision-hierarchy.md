# Provisions nest freely instead of fixed chapter/section/subsection fields

A Provision records its parent and its exact full code identifier; it has no chapter, section, or subsection columns. An earlier draft modelled fixed levels, which would require a value at every level even when a Source does not have one, so levels would be invented. We accepted that "which chapter is this in" is answered by walking parents rather than reading a column.

Consequence: a Provision's parent must belong to the same Source version, and nothing in the schema asserts how deep a hierarchy goes.
