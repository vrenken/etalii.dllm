# Constructive technology assessment of a deterministic LLM

This note does two things. First, it reconstructs from the academic literature how constructive technology
assessment (CTA) studies are carried out: where the approach comes from, its concepts, the steps of a study and the
critiques. Second, it applies that method to EtAlii.Dllm: who the actors around deterministic LLM inference are, what
socio-technical effects bit-exact reproducibility can be expected to have, which futures are plausible, and what
that means for the design of this project now, while changes are still cheap.

It is a first, desk-based round of CTA. A full CTA also brings the actors together in workshops; section 9 gives
the protocol for that round, and section 10 says what this desk study can and cannot claim.

## Contents

1. [What CTA is](#1-what-cta-is)
2. [CTA among the other forms of technology assessment](#2-cta-among-the-other-forms-of-technology-assessment)
3. [How a CTA study is conducted](#3-how-a-cta-study-is-conducted)
4. [Critiques and limits of CTA](#4-critiques-and-limits-of-cta)
5. [Applying CTA to EtAlii.Dllm: object and moment](#5-applying-cta-to-etaliidllm-object-and-moment)
6. [Field mapping: actors, expectations and dynamics](#6-field-mapping-actors-expectations-and-dynamics)
7. [Expected socio-technical effects](#7-expected-socio-technical-effects)
8. [Socio-technical scenarios](#8-socio-technical-scenarios)
9. [Implications for the project and a protocol for the next round](#9-implications-for-the-project-and-a-protocol-for-the-next-round)
10. [Limitations and reflexivity](#10-limitations-and-reflexivity)
11. [References](#11-references)

## 1. What CTA is

Constructive technology assessment was developed in the Netherlands in the second half of the 1980s, when the newly
founded Netherlands Organisation for Technology Assessment (NOTA, since 1994 the Rathenau Instituut) looked for a TA
that would not only warn parliament about impacts after the fact, but take part in shaping technology while it was
being developed (Schot, 1992; Rip, Misa & Schot, 1995). The founding texts are the edited volume *Managing Technology
in Society* (Rip, Misa & Schot, 1995) and the programmatic article *The past and future of constructive technology
assessment* (Schot & Rip, 1997). The approach was later developed mostly at the University of Twente, applied to
clean technologies, medical technologies and, at scale, nanotechnology in the Dutch NanoNed programme (Rip & van
Lente, 2013; Konrad, Rip & Greiving, 2017; Robinson, 2024). A systematic review of CTA applications
(Rodríguez-Cardoso, Ballesteros-Ballesteros & Romero-Ospina, 2021) finds most of them in health care, with growing
use in engineering fields such as nanotechnology, robotics and big data.

CTA starts from three observations.

- **The Collingridge dilemma.** Early in a technology's development its effects cannot yet be known, and when they
  are known, the technology is entrenched and hard to change (Collingridge, 1980). CTA's answer is not to predict
  better, but to make the development process itself more anticipatory and more open to learning.
- **The division of labour between promotion and control.** Developers promote technology; others (regulators,
  NGOs, ethicists) control its effects afterwards. CTA calls this a structural cause of poor outcomes and tries to
  bring impact considerations into the design process, by broadening "the aspects and actors" that count (Schot &
  Rip, 1997).
- **Technology and society co-evolve.** Drawing on evolutionary economics and science and technology studies,
  CTA treats technology as shaped by variation and selection in a socio-technical environment. Effects are not
  "impacts" of a finished artefact on society but emerge from that co-evolution (Rip & Kemp, 1998; Geels, 2002).

Schot and Rip (1997) name three process criteria by which to judge CTA: **anticipation** (considering future
effects early), **reflexivity** (actors seeing that their choices shape outcomes and are contingent) and **social
learning** (first-order learning about the technology and its effects; second-order learning about values,
problem definitions and the roles of the actors). They also distinguish three generic strategies: *technology
forcing* (steering by regulation or procurement), *strategic niche management* (protected spaces in which a new
technology and its users learn together; Kemp, Schot & Hoogma, 1998; Schot & Geels, 2008) and *loci for alignment*
(forums, standards bodies and platforms where actors align their agendas).

## 2. CTA among the other forms of technology assessment

| Approach | Main question | Typical output | Relation to CTA |
| --- | --- | --- | --- |
| Classical ("awareness") TA, e.g. the US Office of Technology Assessment | What will the impacts be, and what should policy do? | Reports to parliament | CTA's starting point and foil: impacts are assessed too late and from outside (Grunwald, 2019) |
| Participatory TA (consensus conferences, citizens' panels) | What do citizens think? | Public recommendations | Shares the broadening of actors, but CTA engages the developers themselves (van Est & Brom, 2012) |
| Real-time TA | How can social research run alongside a research programme? | Analyses fed back into R&D | Close cousin, from the US nanotechnology programme (Guston & Sarewitz, 2002) |
| Midstream modulation / STIR | How do lab-level decisions change when a humanist sits in the lab? | Reflexive shifts in research practice | CTA's "insertion" at the level of one lab (Fisher, Mahajan & Mitcham, 2006; Schuurbiers, 2011) |
| Responsible research and innovation (RRI) | How should innovation be governed to be responsible? | Frameworks, policy | Its four dimensions (anticipation, reflexivity, inclusion, responsiveness) come largely from CTA (Stilgoe, Owen & Macnaghten, 2013; Owen, Macnaghten & Stilgoe, 2012) |
| Ethical CTA | How do technologies mediate values, and how can design address that? | Ethical scenarios, design input | Adds the co-evolution of technology and morality (Kiran, Oudshoorn & Verbeek, 2015; Boenink, Swierstra & Stemerding, 2010; Swierstra & Rip, 2007) |

For a software project the practical difference is this: classical TA would ask a regulator what to do about
deterministic LLMs; CTA asks the builders of one, while there is still time, what their design choices are likely to
set in motion and with whom they should learn.

## 3. How a CTA study is conducted

Published CTA studies on emerging technologies (for example van Merkerk & Robinson, 2006; Rip & te Kulve, 2008;
Robinson, 2009; Parandian, 2012; te Kulve & Rip, 2011; Rip & Robinson, 2013) follow a recognisable sequence. CTA is
not a fixed protocol, and the literature presents these as phases of work rather than a checklist.

### 3.1 Characterise the emerging field

The CTA analyst maps the dynamics of the domain on three levels (van Merkerk & Robinson, 2006): **expectations**
(what is promised, by whom, in which statements), **agendas** (what actors plan to do: research programmes,
roadmaps, product plans) and **networks** (who works with whom, which alliances and supply chains form). Sources are
publications, patents, roadmaps, conference programmes, press releases and interviews. The purpose is to find the
**emerging irreversibilities**: the choices that are becoming path-dependent and will constrain later choices.

Two concepts from the sociology of expectations are central here (van Lente, 1993; Borup et al., 2006):

- **Promise-requirement cycles.** A promise ("this technology will make X possible"), once accepted, becomes a
  requirement that actors must meet, and so shapes agendas (van Lente & Rip, 1998).
- **Waiting games.** Actors each wait for others to move first (users wait for proven products, developers for
  demand), which can stall a promising technology (Parandian, Rip & te Kulve, 2012).

The analyst also distinguishes **enactors**, who build the technology and see it from inside, from
**comparative selectors**, who choose among alternatives from outside: users, investors, regulators (Garud &
Ahlstrom, 1997; Rip, 2006a). Enactors tend to see effects as someone else's problem; selectors tend to see the
technology as a black box. A good part of CTA's work is making the two perspectives meet. Enactors' own
"folk theories" (for example "the public will accept it once they understand it") are part of what gets examined
(Rip, 2006b).

### 3.2 Identify endogenous futures and build socio-technical scenarios

From the field analysis the analyst derives **endogenous futures**: futures that grow out of the dynamics already
visible, rather than wished-for end states (Rip & te Kulve, 2008). These are written up as
**socio-technical scenarios**: narratives that follow a technology through **branching points** where actors'
choices, events or external shifts send it one way or another, and that show how effects arise from the
interactions of several actors over time (Rip & te Kulve, 2008; Robinson, 2009, who calls them "co-evolutionary
scenarios"). Scenarios are framed with the multi-level perspective: a niche technology, the prevailing regime it
has to fit into or replace, and the wider landscape of trends and shocks (Geels, 2002). Good scenarios are
plausible, differ in the branching choices they explore, and are specific enough that actors recognise themselves
in them.

### 3.3 Bring actors together: strategy articulation workshops and bridging events

The scenarios serve as the input to workshops that bring together enactors and selectors who do not normally meet
(Robinson, 2009; Parandian, 2012). Parandian (2012) calls these **bridging events**. The workshops are prepared with
**pre-engagement**: interviews and tailored scenario documents that make participants ready to engage productively
(te Kulve & Rip, 2011). In the workshop participants articulate their strategies in the light of the scenarios and
of each other's positions. The aim is not consensus but better-informed strategies ("strategy articulation") and
social learning (Rip & Robinson, 2013).

### 3.4 Insertion and evaluation

CTA analysts work from an **insider-outsider** position: close enough to the technology's development to be
relevant and heard, independent enough to bring in other perspectives. Rip and Robinson (2013) call this the
**methodology of insertion**. The outcome is judged by the criteria of section 1: did actors anticipate more
broadly, become more reflexive about their choices, and learn at the first and second order? Evaluation uses
observation of workshops, follow-up interviews and changes in actors' agendas.

### 3.5 Summary of the method

| Phase | Activities | Output |
| --- | --- | --- |
| 1. Field characterisation | Map expectations, agendas and networks; enactors and selectors; irreversibilities | Field analysis |
| 2. Scenarios | Endogenous futures, branching points, multi-level framing | Two to four socio-technical scenarios |
| 3. Pre-engagement | Interviews, tailored scenario documents | Prepared participants |
| 4. Bridging events | Strategy articulation workshops | Articulated strategies, joint agenda items |
| 5. Feedback and evaluation | Feed results into design and agendas; assess learning | Design changes, evaluation |

## 4. Critiques and limits of CTA

The literature raises five recurring objections, which this note tries to take into account.

- **Limited public participation.** CTA engages actors with a stake in the technology's development; citizens and
  affected but unorganised groups are often absent (Genus & Coles, 2005). Genus (2006) argues that CTA should be
  reframed as democratic, reflexive discourse, not only strategy articulation among incumbents.
- **Closing down versus opening up.** An appraisal can "close down" on a single recommendation or "open up" a range
  of options and the assumptions behind them (Stirling, 2008). Strategy articulation workshops risk closing down
  around the enactors' framing.
- **Capture by the insiders.** The insider-outsider position can slide into advocacy for the technology (Rip &
  Robinson, 2013); independence has to be worked at.
- **Ethics treated as an afterthought.** Classical CTA focused on socio-technical dynamics; ethical CTA argues that
  values are changed by technologies too and belong in the scenarios (Kiran, Oudshoorn & Verbeek, 2015).
- **Humility.** Anticipation can be mistaken for prediction. Jasanoff (2003) asks for "technologies of humility":
  approaches that make visible what is uncertain, who is vulnerable and how learning happens.

## 5. Applying CTA to EtAlii.Dllm: object and moment

**Object.** The object of this assessment is not "LLMs" in general, whose effects are extensively catalogued
(Bender et al., 2021; Bommasani et al., 2021; Weidinger et al., 2022), but the specific property this project adds:
*bit-exact reproducibility of LLM output on the same hardware, whatever the load, batching or thread scheduling*
(see [deterministic inference](deterministic-inference.md)). The question is what changes in the world when that
property becomes available, and what it means for design choices here.

**Moment.** In multi-level terms, EtAlii.Dllm is a niche: an open-source engine for small imported models (up to
about 1.5B parameters today), run on CPU or CUDA with fixed-order kernels, speaking the OpenAI, Anthropic and Ollama
APIs and MCP (see the [README](../../README.md)). The regime it interfaces with is mainstream LLM serving, in which
output is not reproducible and determinism is documented at best as "best effort". That makes this the early side of
the Collingridge dilemma: the design (what the fingerprint covers, how ids are derived, what defaults are chosen)
is still cheap to change, while effects can only be anticipated, not observed.

## 6. Field mapping: actors, expectations and dynamics

### 6.1 Expectations and agendas in the domain (desk analysis)

| When | Development | What it shows |
| --- | --- | --- |
| 2023 | OpenAI adds a `seed` parameter and `system_fingerprint`, documented as best-effort determinism | The regime acknowledges demand for reproducibility but does not promise it |
| 2024 | Atil et al. measure accuracy varying by up to 15% between runs of LLMs configured to be deterministic, and suggest the cause is tied to efficient batching | Reproducibility becomes a scientific-validity problem for evaluations |
| 2025 | Thinking Machines Lab (He, 2025) shows that batch-variant kernels, not GPU "randomness", cause most serving non-determinism, and that batch-invariant kernels fix it at moderate cost; vLLM and SGLang add deterministic or batch-invariant modes | The promise "reproducible serving is feasible" is made credibly by a regime actor |
| 2025 | Gu et al. show that prompt caches shared across users in commercial APIs leak which prompts others sent, through timing | Serving optimisations that make requests interact have privacy consequences |
| 2024 to 2027 | The EU AI Act (Regulation (EU) 2024/1689) phases in obligations for logging (Art. 12), transparency (Art. 13), human oversight including awareness of automation bias (Art. 14) and accuracy and robustness (Art. 15) for high-risk systems, and documentation for general-purpose models (Art. 53) | Selectors in regulated sectors acquire requirements that reproducibility can help meet |
| 2026 | EtAlii.Dllm: run-to-run and batch-invariant bit-exactness as a hard requirement, including fine-tuning, prompt caching and concurrent batching, checked by golden hashes in CI | A niche actor makes the strong version of the promise |

The emerging **promise** is "deterministic inference makes LLMs auditable, testable and trustworthy". Following the
promise-requirement cycle (van Lente & Rip, 1998), once accepted, it will turn into requirements: auditors asking for
replay, evaluation standards asking for reproducible runs. It also risks sliding from "reproducible" to "reliable",
which are different properties (section 7).

The **irreversibilities** now forming are: (a) which layer owns determinism (the engine, as here, or the API
contract); (b) what a fingerprint identifies (weights only, or weights plus engine and hardware); (c) whether
reproducibility is scoped to one machine, as here, or expected across machines; (d) whether determinism is a default
or an opt-in mode with a performance cost, as in vLLM and SGLang.

A likely **waiting game**: regulated users wait for a mature, fast, supported deterministic stack before building
compliance processes on replay; builders of such stacks wait for demonstrated demand before investing in
performance and hardware breadth.

### 6.2 Actors

| Actor | Enactor or selector | Stake in deterministic inference | Likely concern |
| --- | --- | --- | --- |
| EtAlii.Dllm maintainers (including the Claude sessions that develop it) | Enactor | The core property of the project | Performance, model coverage; folk theory that "determinism is simply good" |
| Mainstream serving engines (vLLM, SGLang, llama.cpp) and hardware vendors | Enactors (regime) | Deterministic modes as a feature, at a throughput cost | Keeping the fast path fast; kernel autotuning versus fixed paths |
| Model developers (e.g. the SmolLM2 and Qwen teams) | Enactors upstream | Their models become replayable objects | Licence and attribution of imported weights |
| Application and agent builders (via OpenAI/Anthropic/Ollama APIs and MCP) | Selectors | Regression tests with exact expected output; cacheable answers | Speed, model quality; the same answer to every user |
| Evaluators and ML researchers | Selectors | Reproducible benchmark runs (Atil et al., 2024; Gundersen & Kjensmo, 2018; Pineau et al., 2021) | Results valid only for the pinned hardware |
| Compliance officers and auditors in regulated sectors (finance, health, public sector) | Selectors | Replaying an incident from a log | Auditors rarely have the operator's hardware |
| Regulators and standards bodies | Selectors (technology forcing) | Logging, robustness and oversight obligations | Reproducibility mistaken for accuracy; compliance theatre |
| End users and affected people (not organised) | Absent selectors | Answers that can be verified later; consistency | Privacy of prompts; everyone receiving the same flawed answer |
| Adversaries (jailbreakers, prompt-injection authors) | Unintended users | An attack that works once works every time | (Their success is others' concern) |

End users and affected people are in the table to mark their absence from the field: nobody organises them around
this property, which is exactly the gap Genus and Coles (2005) point out in CTA practice.

## 7. Expected socio-technical effects

Effects are listed with the mechanism through which they would arise, who gains and who bears the cost, and a
qualitative confidence based on how directly the mechanism follows from the engine's behaviour. They are
anticipations for discussion, not findings.

| # | Effect | Mechanism | Gains / bears | Confidence |
| --- | --- | --- | --- | --- |
| E1 | **Replayable incidents and audits** | A logged request (weights fingerprint, prompt, parameters) replays to the same tokens, so an operator or auditor can reproduce exactly what the model said | Operators, auditors, affected people seeking redress | High on the same hardware; low across hardware (see E8) |
| E2 | **Exact regression tests for LLM applications** | Applications can assert exact outputs, as this repository does with golden hashes | Builders; quality assurance | High |
| E3 | **Reproducible evaluations** | Benchmark scores stop varying run to run | Researchers, evaluators | High, for the models the engine supports |
| E4 | **Non-repudiation, in both directions** | Given the model and request, anyone can verify what the output was, and a claimed output can be checked against a claimed prompt | Parties in disputes; journalism and fact-checking | Medium; depends on logging and on the model file being available |
| E5 | **Prompt confirmation and linkability** | Response ids are hashes of the model fingerprint and the request ([api.md](../api.md#determinism-guarantees)), so anyone who can guess a request can confirm it from an id, and identical requests from different users carry the same id. With a shared prompt cache, the `usage` cache counters reveal whether an identical prefix was sent before, a mechanism shown in commercial APIs by Gu et al. (2025) | Users bear it | High that the mechanism exists; impact depends on deployment (single-user local use is unaffected) |
| E6 | **Reliable attacks** | A jailbreak or prompt injection that succeeds once succeeds on every retry, and can be shared as a reproducible recipe; equally, defenders get exact regression tests for fixes | Attackers and defenders both | Medium |
| E7 | **"Determinism-washing" and automation bias** | Consistency reads as correctness; a reproducible wrong answer looks more trustworthy than a varying one, feeding the over-reliance that Art. 14 of the AI Act asks oversight to guard against (see also Parasuraman & Manzey, 2010) | End users bear it | Medium; it depends on how the property is communicated |
| E8 | **Hardware-bound evidence** | The guarantee is scoped to the same hardware; an auditor on different hardware may not be able to replay, and archived evidence ages with the hardware | Auditors, archivists bear it | Medium; today CI happens to agree across Linux, Windows and macOS, but that is not promised |
| E9 | **Outcome homogenisation** | Greedy decoding (the default, temperature 0) and a fixed default seed give every user the same answer to the same prompt; where many decision-makers use the same model this amplifies monoculture effects (Kleinberg & Raghavan, 2021; Bommasani et al., 2022) | Individuals who are systematically on the wrong side of an answer bear it | Medium for aggregate effects, which require wide deployment |
| E10 | **Cheaper serving through exact caching** | Identical requests can be answered from cache without any quality trade-off | Operators; energy use | High technically, modest economically at small-model scale |
| E11 | **Niche positioning by cost** | The fixed-order, double-accumulator kernels cost throughput, so the engine suits small models, verification and regulated niches rather than bulk serving | Shapes who adopts it | High |
| E12 | **Reproducible fine-tuning and provenance** | Equal training runs write byte-identical models ([training](../training.md)), so a model's lineage (base weights, data order, hyperparameters) can be verified by recomputation | Researchers, auditors, model licensors | High on the same hardware |

Three patterns stand out. First, most benefits (E1 to E4, E12) accrue to organised selectors: operators, auditors,
researchers. Most costs (E5, E7, E9) fall on end users, the actor group that is absent from the field. That is the
classic CTA signal that the missing perspective has to be brought in deliberately. Second, several effects are
**double-edged by the same mechanism**: the determinism that makes an incident replayable (E1) makes an attack
replayable (E6), and the content-derived ids that make responses verifiable (E4) make prompts confirmable (E5).
Third, the strongest promise, auditability, is weakened by the project's own scoping decision (E8): "same hardware"
was chosen to allow fast hardware-specific kernels, but the auditor is the one actor who, by definition, is not on
the operator's hardware.

## 8. Socio-technical scenarios

Following Rip and te Kulve (2008), each scenario follows the niche through branching points to about 2030. They are
endogenous futures: each continues a dynamic from section 6.1. They are not predictions, and more than one can
partly come true.

### Scenario A: "Reference engine" (mainstream absorbs the property)

Mainstream engines make their batch-invariant modes faster until determinism is a cheap switch in vLLM and SGLang.
*Branching point:* whether the overhead drops below what operators notice. If it does, EtAlii.Dllm stops being the
only way to get reproducible output and becomes a **reference**: a small, readable engine whose documented
evaluation orders and golden hashes are used to check other engines. Selectors adopt determinism quietly, as a
default; effect E9 (homogenisation) spreads with it, while E1 to E3 become common. The project's value moves from
serving to verification and teaching.

### Scenario B: "Regulatory pull" (replay becomes a compliance artefact)

Standards for high-risk AI systems under the AI Act come to treat exact replay of logged decisions as good practice
for logging and robustness. *Branching point:* whether standards specify replay in terms of a reproducible
environment (weights, engine version, hardware class) or merely require logs. If the former, deployers in regulated
sectors look for engines with a documented determinism domain, and the promise-requirement cycle turns
EtAlii.Dllm's golden-hash practice into a template. The risk in this scenario is compliance theatre (E7): replay
proves consistency, not correctness, and a system can be perfectly reproducible and still wrong. A second
*branching point* comes when an auditor cannot reproduce a result on their own hardware (E8), which puts pressure on
engines to offer cross-hardware reproducibility for audit, the level 3 that this project deliberately does not
promise.

### Scenario C: "Contested determinism" (the privacy and security side dominates)

An incident makes the double edge visible: prompts confirmed from logged response ids (E5), a shared cache leaking
which documents colleagues queried, or a widely shared jailbreak that "always works" (E6). *Branching point:* how
operators respond. Either they turn determinism into an opt-in mode, with per-deployment secrets in ids and
per-user seeds, keeping replay for those who hold the secret; or they abandon determinism in shared deployments and
keep it for local and evaluation use. In the first branch reproducibility becomes a privilege of the operator and
auditor rather than a public property.

### Scenario D: "Local and private" (the niche stays a niche)

Model families and hardware move faster than a from-scratch engine can follow (Phase 9 of the roadmap is exactly
this race). *Branching point:* whether the project keeps pace with mainstream model families. If it does not, the
engine settles in a niche of local, single-user and research use: reproducible personal assistants, classroom and
lab use, reproducible experiments. Here E5 and E9 matter little (one user, one machine), E3 and E12 matter most,
and the waiting game of section 6.1 is never resolved for regulated users.

| | A: Reference | B: Regulatory pull | C: Contested | D: Local niche |
| --- | --- | --- | --- | --- |
| Main driver | Regime catches up | Technology forcing | Incident, public concern | Pace of model families |
| Dominant effects | E1 to E3 widely, E9 | E1, E7, E8 | E5, E6 | E3, E12 |
| What the project most needs | Clear documentation of evaluation orders | A determinism domain in the fingerprint; portable mode | Keyed ids, per-user seeds, cache isolation | Model coverage, ease of use |

## 9. Implications for the project and a protocol for the next round

### 9.1 Design implications ("constructive" output)

The recommendations below are what the scenarios have in common: choices that are cheap now, help in more than one
scenario and hurt in none. None is implemented by this note; each would be its own issue and pull request.

1. **Make the determinism domain explicit.** `system_fingerprint` identifies the weights. A replay also depends on the
   engine version, the kernel path chosen (scalar, SSE2, AVX2, NEON, CUDA) and quantisation. Reporting these, next to
   the fingerprint or in a separate field, lets an auditor know whether a replay should match (E1, E8; scenarios B, A).
2. **Turn today's accidental cross-platform agreement into an opt-in guarantee.** CI shows Linux, Windows and macOS
   agree today. A documented "portable" mode, kept by a CI check, would give auditors a way to replay on their own
   hardware without giving up the fast hardware-specific paths the scoping decision was made for (E8; scenario B).
3. **Offer keyed ids for shared deployments.** Deriving ids from an HMAC with a per-deployment secret instead of a
   plain hash keeps ids deterministic for the operator (and for auditors given the key) while stopping third parties
   from confirming prompts or linking users (E5; scenario C). The current behaviour stays the default for local use.
4. **Isolate the prompt cache per tenant, or hide its counters, in multi-user deployments.** `--prompt-cache 0`
   already removes the counters; documenting the side channel and offering per-key caches addresses the mechanism
   Gu et al. (2025) found in commercial APIs (E5).
5. **Say plainly that reproducible is not correct.** The README and getting-started guide should state that
   determinism makes errors reproducible, not rarer, so the property is not read as a quality claim (E7; scenario B).
6. **Document the homogenisation trade-off and the seed as a design lever.** Temperature 0 and `seed` 0 as defaults
   are the right choice for testing; deployments serving many people may want per-user seeds, still reproducible
   per user (E9).
7. **Provide replay tooling.** A `dllm replay` that takes a logged request and checks the output against its
   hashes would make E1 and E12 usable by people other than the maintainers.
8. **Treat adversarial reproducibility as a test asset.** Keep known jailbreak and injection prompts as golden
   regression tests for any safety measures (E6), so the property that helps attackers helps defenders at least as
   much.

### 9.2 Protocol for a strategy articulation workshop

The desk study covers phases 1 and 2 of section 3.5. The next round would add phases 3 to 5:

- **Participants** (8 to 14): two maintainers; one developer from a mainstream serving engine; two application or
  agent builders; one ML evaluator; one compliance officer or auditor from a regulated sector; one privacy or
  security researcher; one representative of end users or a digital-rights organisation (to counter the absence
  noted in section 6.2); one CTA facilitator who is not a maintainer.
- **Pre-engagement:** a 30-minute interview with each participant on their expectations and concerns; a two-page
  version of the scenarios adapted to their role (te Kulve & Rip, 2011).
- **Agenda (one day):** the field analysis and effects table; each scenario played through its branching points in
  mixed groups, asking "what would you do at this point, and what would you need from the others?"; strategy
  articulation per actor group; a plenary on joint agenda items; explicit time for dissent and for options the
  scenarios missed (opening up rather than closing down; Stirling, 2008).
- **Evaluation:** a follow-up after three months on changes in participants' agendas and in the project's roadmap,
  judged against anticipation, reflexivity and social learning (Schot & Rip, 1997).
- **Lightweight alternative:** where a physical workshop is not feasible, the same scenarios as a GitHub Discussion
  per scenario, with invited participants from each actor group.

## 10. Limitations and reflexivity

- **Desk-based.** The field analysis relies on publications and public documentation, without interviews. Actor
  concerns in section 6.2 are inferred and have to be tested in the next round.
- **Insider position.** The author of this note is one of the Claude sessions that develops EtAlii.Dllm, a pure
  enactor. CTA's insider-outsider balance (Rip & Robinson, 2013) is therefore tilted towards the inside, and the
  most important check on this note is review by people outside the project. The effects table deliberately includes
  effects that argue against the project's current choices (E5, E7, E8, E9) to counter that tilt.
- **Scope.** The assessment covers the reproducibility property, not LLMs as such; general LLM harms (bias,
  misinformation, environmental cost) are covered by the literature cited in section 5 and apply here as elsewhere.
- **Scenarios are not forecasts.** Their value lies in the branching points they make discussable, not in their
  probability.

## 11. References

- Atil, B., et al. (2024). Non-determinism of "deterministic" LLM settings. *arXiv:2408.04667*.
- Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can
  language models be too big? *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency
  (FAccT)*, 610–623.
- Boenink, M., Swierstra, T., & Stemerding, D. (2010). Anticipating the interaction between technology and morality:
  A scenario study of experimenting with humans in bionanotechnology. *Studies in Ethics, Law, and Technology, 4*(2).
- Bommasani, R., Hudson, D. A., Adeli, E., et al. (2021). On the opportunities and risks of foundation models.
  *arXiv:2108.07258*.
- Bommasani, R., Creel, K. A., Kumar, A., Jurafsky, D., & Liang, P. (2022). Picking on the same person: Does
  algorithmic monoculture lead to outcome homogenization? *Advances in Neural Information Processing Systems 35
  (NeurIPS 2022)*.
- Borup, M., Brown, N., Konrad, K., & van Lente, H. (2006). The sociology of expectations in science and technology.
  *Technology Analysis & Strategic Management, 18*(3–4), 285–298.
- Collingridge, D. (1980). *The Social Control of Technology*. London: Frances Pinter.
- European Union (2024). Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence
  (Artificial Intelligence Act). *Official Journal of the European Union*, L, 12.7.2024.
- Fisher, E., Mahajan, R. L., & Mitcham, C. (2006). Midstream modulation of technology: Governance from within.
  *Bulletin of Science, Technology & Society, 26*(6), 485–496.
- Garud, R., & Ahlstrom, D. (1997). Technology assessment: A socio-cognitive perspective. *Journal of Engineering and
  Technology Management, 14*(1), 25–48.
- Geels, F. W. (2002). Technological transitions as evolutionary reconfiguration processes: A multi-level
  perspective and a case-study. *Research Policy, 31*(8–9), 1257–1274.
- Genus, A. (2006). Rethinking constructive technology assessment as democratic, reflective, discourse.
  *Technological Forecasting and Social Change, 73*(1), 13–26.
- Genus, A., & Coles, A.-M. (2005). On constructive technology assessment and limitations on public participation
  in technology assessment. *Technology Analysis & Strategic Management, 17*(4), 433–443.
- Grunwald, A. (2019). *Technology Assessment in Practice and Theory*. London: Routledge.
- Gu, C., Li, X. L., Kuditipudi, R., Liang, P., & Hashimoto, T. (2025). Auditing prompt caching in language model
  APIs. *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. arXiv:2502.07776.
- Gundersen, O. E., & Kjensmo, S. (2018). State of the art: Reproducibility in artificial intelligence. *Proceedings
  of the AAAI Conference on Artificial Intelligence, 32*(1).
- Guston, D. H., & Sarewitz, D. (2002). Real-time technology assessment. *Technology in Society, 24*(1–2), 93–109.
- He, H., & Thinking Machines Lab (2025). Defeating nondeterminism in LLM inference. *Thinking Machines Lab:
  Connectionism*.
- Jasanoff, S. (2003). Technologies of humility: Citizen participation in governing science. *Minerva, 41*(3),
  223–244.
- Kemp, R., Schot, J., & Hoogma, R. (1998). Regime shifts to sustainability through processes of niche formation:
  The approach of strategic niche management. *Technology Analysis & Strategic Management, 10*(2), 175–198.
- Kiran, A. H., Oudshoorn, N., & Verbeek, P.-P. (2015). Beyond checklists: Toward an ethical-constructive technology
  assessment. *Journal of Responsible Innovation, 2*(1), 5–19.
- Kleinberg, J., & Raghavan, M. (2021). Algorithmic monoculture and social welfare. *Proceedings of the National
  Academy of Sciences, 118*(22), e2018340118.
- Konrad, K., Rip, A., & Greiving, V. C. S. (2017). Constructive technology assessment: STS for and with technology
  actors. *EASST Review, 36*(3).
- Owen, R., Macnaghten, P., & Stilgoe, J. (2012). Responsible research and innovation: From science in society to
  science for society, with society. *Science and Public Policy, 39*(6), 751–760.
- Parandian, A. (2012). *Constructive TA of newly emerging technologies: Stimulating learning by anticipation
  through bridging events* (Doctoral dissertation). Delft University of Technology.
- Parandian, A., Rip, A., & te Kulve, H. (2012). Dual dynamics of promises, and waiting games around emerging
  nanotechnologies. *Technology Analysis & Strategic Management, 24*(6), 565–582.
- Parasuraman, R., & Manzey, D. H. (2010). Complacency and bias in human use of automation: An attentional
  integration. *Human Factors, 52*(3), 381–410.
- Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021). Improving reproducibility in machine learning research
  (a report from the NeurIPS 2019 Reproducibility Program). *Journal of Machine Learning Research, 22*(164), 1–20.
- Rip, A. (2006a). A co-evolutionary approach to reflexive governance, and its ironies. In J.-P. Voß, D. Bauknecht &
  R. Kemp (Eds.), *Reflexive Governance for Sustainable Development* (pp. 82–100). Cheltenham: Edward Elgar.
- Rip, A. (2006b). Folk theories of nanotechnologists. *Science as Culture, 15*(4), 349–365.
- Rip, A., & Kemp, R. (1998). Technological change. In S. Rayner & E. L. Malone (Eds.), *Human Choice and Climate
  Change, Vol. 2: Resources and Technology* (pp. 327–399). Columbus, OH: Battelle Press.
- Rip, A., Misa, T. J., & Schot, J. (Eds.) (1995). *Managing Technology in Society: The Approach of Constructive
  Technology Assessment*. London: Pinter.
- Rip, A., & Robinson, D. K. R. (2013). Constructive technology assessment and the methodology of insertion. In
  N. Doorn, D. Schuurbiers, I. van de Poel & M. E. Gorman (Eds.), *Early Engagement and New Technologies: Opening Up
  the Laboratory* (pp. 37–53). Dordrecht: Springer.
- Rip, A., & te Kulve, H. (2008). Constructive technology assessment and socio-technical scenarios. In E. Fisher,
  C. Selin & J. M. Wetmore (Eds.), *The Yearbook of Nanotechnology in Society, Vol. 1: Presenting Futures*
  (pp. 49–70). Dordrecht: Springer.
- Rip, A., & van Lente, H. (2013). Bridging the gap between innovation and ELSA: The TA program in the Dutch Nano-R&D
  program NanoNed. *NanoEthics, 7*(1), 7–16.
- Robinson, D. K. R. (2009). Co-evolutionary scenarios: An application to prospecting futures of the responsible
  development of nanotechnology. *Technological Forecasting and Social Change, 76*(9), 1222–1239.
- Robinson, D. K. R. (2024). Constructive technology assessment: Supporting the reflexive co-evolution of technology
  and society. In A. Grunwald (Ed.), *Handbook of Technology Assessment* (ch. 27). Cheltenham: Edward Elgar.
- Rodríguez-Cardoso, Ó.-I., Ballesteros-Ballesteros, V.-A., & Romero-Ospina, M.-F. (2021). Constructive technology
  assessment: Systematic review and future study needs. *Revista Facultad de Ingeniería, 30*(55), e12459.
- Schot, J. (1992). Constructive technology assessment and technology dynamics: The case of clean technologies.
  *Science, Technology, & Human Values, 17*(1), 36–56.
- Schot, J., & Geels, F. W. (2008). Strategic niche management and sustainable innovation journeys: Theory, findings,
  research agenda, and policy. *Technology Analysis & Strategic Management, 20*(5), 537–554.
- Schot, J., & Rip, A. (1997). The past and future of constructive technology assessment. *Technological
  Forecasting and Social Change, 54*(2–3), 251–268.
- Schuurbiers, D. (2011). What happens in the lab: Applying midstream modulation to enhance critical reflection in
  the laboratory. *Science and Engineering Ethics, 17*(4), 769–788.
- Stilgoe, J., Owen, R., & Macnaghten, P. (2013). Developing a framework for responsible innovation. *Research
  Policy, 42*(9), 1568–1580.
- Stirling, A. (2008). "Opening up" and "closing down": Power, participation, and pluralism in the social appraisal
  of technology. *Science, Technology, & Human Values, 33*(2), 262–294.
- Swierstra, T., & Rip, A. (2007). Nano-ethics as NEST-ethics: Patterns of moral argumentation about new and
  emerging science and technology. *NanoEthics, 1*(1), 3–20.
- te Kulve, H., & Rip, A. (2011). Constructing productive engagement: Pre-engagement tools for emerging
  technologies. *Science and Engineering Ethics, 17*(4), 699–714.
- van Est, R., & Brom, F. (2012). Technology assessment, analytic and democratic practice. In R. Chadwick (Ed.),
  *Encyclopedia of Applied Ethics* (2nd ed., Vol. 4, pp. 306–320). San Diego: Academic Press.
- van Lente, H. (1993). *Promising Technology: The Dynamics of Expectations in Technological Developments*
  (Doctoral dissertation). University of Twente. Delft: Eburon.
- van Lente, H., & Rip, A. (1998). The rise of membrane technology: From rhetorics to social reality. *Social
  Studies of Science, 28*(2), 221–254.
- van Merkerk, R. O., & Robinson, D. K. R. (2006). Characterizing the emergence of a technological field:
  Expectations, agendas and networks in lab-on-a-chip technologies. *Technology Analysis & Strategic Management,
  18*(3–4), 411–428.
- Weidinger, L., Uesato, J., Rauh, M., et al. (2022). Taxonomy of risks posed by language models. *Proceedings of the
  2022 ACM Conference on Fairness, Accountability, and Transparency (FAccT)*, 214–229.
