# Civil code research

An assistant that helps a civil engineer find which governing codes and criteria manuals apply to a stated project situation and what they say, concluding only from evidence it can cite. The engineer remains responsible for any determination made from the answer.

## Language

### People and questions

**Engineer**:
The civil engineer asking a research question and responsible for any determination made from the answer.
_Avoid_: User, client, customer

**Conversation**:
The ordered exchange between an Engineer and the assistant. It carries the Project context that accumulates across questions.
_Avoid_: Session, chat thread, project

**Research question**:
A question about what governing sources require or say, asked in order to find and cite authority.
_Avoid_: Query, prompt, search

**Project context**:
The facts about a project that decide which Authority governs it (above all its city), which Regime applies within that Authority, and the dates that fix which edition applies.
_Avoid_: Project record, metadata, profile

### Authorities and sources

**Authority**:
A body whose adopted rules can govern a project, such as a city.
_Avoid_: Agency, owner

**Jurisdiction**:
The geographic or administrative area an Authority governs.
_Avoid_: Region, location (a project has a location; an Authority has a jurisdiction)

**Source**:
A published document issued by an Authority, such as a criteria manual, code, or amendment.
_Avoid_: Document, manual, file, PDF

**Source version**:
One specific edition of a Source, identified by the date it took effect, with its own exact text. Its text never changes once registered; a newer effective edition becomes a new Source version and is parsed again.
_Avoid_: Edition (an attribute of a version), revision, document version

**Capture date**:
The date a Source version's file was uploaded to the system, kept to track how current the stored copy is and to spot drift. It is not the effective date.
_Avoid_: Version date, edition date

**Provision**:
An addressable unit of a Source version, such as a section or clause, identified by its full code identifier and possibly nested inside another Provision.
_Avoid_: Section, chapter, clause (these name one level only), chunk

**Applicability**:
Whether a Source version governs a particular project. Distinct from topical relevance: a relevant passage from the wrong edition is not applicable.
_Avoid_: Relevance, match

**Selected scope**:
The Sources a question is focused on, either chosen by the Engineer or inferred from how the question names them (for example "in the Drainage Manual"). It narrows the search; it does not establish Applicability.
_Avoid_: Filter, context

### Following the text

**Definition**:
A Provision that fixes the meaning of a term used elsewhere, so that the term cannot be read without it.
_Avoid_: Glossary entry

**Amendment**:
A change issued by an Authority to a Provision or a Source version, which can alter what the original text requires.
_Avoid_: Update, revision, errata

**Reference**:
A pointer from one Provision to another Provision, Source, or outside standard that the reader must follow to understand the requirement.
_Avoid_: Link, citation (a citation points from an answer to Evidence)

### Evidence and outcomes

**Evidence**:
Verbatim text from a Provision of an applicable Source version, together with where it came from.
_Avoid_: Context, passage, snippet, chunk

**Authorized evidence**:
Evidence from an applicable Source version that the system is permitted to use.
_Avoid_: Approved sources, trusted data

**Web evidence**:
Text from an official government web page, cited by its URL, that is not tied to any registered Source version.
_Avoid_: Web source, search result, Evidence (unqualified)

**Research answer**:
A response that states a Conclusion, or what Evidence or Web evidence says, with every statement traceable to specific cited text and the response labelled by which of the two it rests on.
_Avoid_: Answer (unqualified), guidance, recommendation

**Conclusion**:
A statement the assistant draws from Evidence, such as which design zone applies to a project, always shown with the Provisions it rests on.
_Avoid_: Determination (reserved for the Engineer's professional judgment), finding

**Regime**:
A set of requirements that applies only under stated conditions, such as a named zone, so that the same question can have different answers under different Regimes.
_Avoid_: Zone (one kind of Regime), case, scenario

**Answer review**:
An independent judgment, by a different model from the one that wrote the answer, of whether a Research answer is supported by the Provisions it cites, with a confidence score recorded for monitoring and never shown to the Engineer as a probability of correctness.
_Avoid_: Grading, verification, validation

**Clarification**:
A question back to the Engineer because the Project context or the Research question is not sufficient to proceed, or because more than one Regime could apply; it names the Regimes and gives no value until the Engineer says which applies.
_Avoid_: Follow-up question, prompt for more info

**Evidence gap**:
A statement that the applicable Source versions contain no sufficient Evidence for the question, naming what was searched.
_Avoid_: No results, unknown

**Escalation**:
Referring a question to a qualified person for determination because the Evidence conflicts, is ambiguous, or the decision is not the system's to make.
_Avoid_: Refusal, fallback

**Research outcome**:
The terminal result of a Research question: a Research answer, a Clarification, an Evidence gap, or an Escalation.
_Avoid_: Response type, result

**Blocked message**:
A message stopped by filtering before any research because it is abusive, unsafe, a prompt injection, or off-topic; it is logged and the Engineer is told it cannot be processed.
_Avoid_: Refusal, invalid request

### Routing

**Route**:
The classification of an incoming request as retrieval, conversation, or invalid, made before any research is done.
_Avoid_: Intent, category

**Routing run**:
The saved record of one Route classification, including the exact transcript and Selected scope it saw and the policy and model used.
_Avoid_: Trace, log entry

**Human evaluation**:
A reviewer's judgment of what the Route should have been, recorded alongside the model's decision and never replacing it.
_Avoid_: Correction, feedback, label override
