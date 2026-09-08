# CB16 R11 Stage-4 S4C — One-Time Authority Adoption & Lineage Preservation R0

## Scope

S4C implements one operation only: bind an already accepted CB16 authority state to the
canonical R11 runtime authority identity. Adoption is infrastructure metadata. It is not
training, replay, Evidence mint/admission, a scientific verdict, a Champion tournament,
or generation advancement.

Highest authority remains the Semantic Freeze and semantic contracts. Existing Python
implementations are reference/runtime mechanisms, not semantic authority.

Scientific status remains:

`DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`

## Trusted input boundary

The adoption API deliberately separates:

1. `AuthorityAdoptionContract.accepted_source`: the exact state already accepted by
   upstream authority and supplied by final integration; and
2. `observed_source`: the state identities observed by the caller at cutover.

Every field must match exactly before any receipt write. S4C does not discover or invent
source identities. In particular it never guesses a Champion, checkpoint, Journal head,
Evidence root, source generation, or Semantic Freeze identity.

The source identity binds:

- source repository and source commit SHA;
- Semantic Freeze Git-blob identity;
- source generation;
- Champion identity and semantic/content hash;
- checkpoint identity and semantic/content hash;
- Evidence root identity;
- authoritative Journal head identity; and
- checkpoint root identity.

The target contract supplies only the canonical R11 authority identity. Adoption
generation is derived from the source generation and must equal it.

## Receipt identity

The receipt schema is `CB16_R11_STAGE4_AUTHORITY_ADOPTION_RECEIPT_V1`.

Its canonical content hash is SHA-256 over canonical JSON containing the complete source
identity, target authority identity, unchanged adoption generation, semantic guards, and
identity rules. JSON is UTF-8, sorted by key, compact, finite-valued, and terminated by one
newline when persisted.

`metadata.adoption_timestamp_utc` is explicitly excluded from the hash identity. Therefore
two otherwise identical attempts at different times have the same adoption identity and
the second attempt is an idempotent `ALREADY_ADOPTED` result rather than a new history.

Required semantic guards are all fail-closed and immutable:

- `scientific_history_rewritten = false`
- `new_evidence_created = false`
- `new_scientific_verdict = false`
- `generation_advanced_by_adoption = false`

## Persistence / crash semantics

The receipt is first serialized completely to a same-directory temporary file and fsynced.
Publication uses a hard-link create-if-absent operation. It never uses replace-over-existing
semantics. If the final path already exists, that receipt must validate and match the exact
accepted contract; otherwise adoption fails closed.

This gives three important outcomes:

- crash before publication: there is no authoritative receipt;
- crash after publication: the final path already contains a complete, verifiable receipt;
- concurrent attempts: exactly one final pathname can be created; a conflicting loser is
  rejected and an identical loser becomes an idempotent success.

If the filesystem does not provide the required atomic create-if-absent hard-link primitive,
S4C fails closed rather than falling back to a weaker overwrite protocol. Final integration
may replace the local store primitive only with an equivalently strong contract.

Temporary siblings are non-authoritative and are never accepted as adoption history.

## No source mutation capability

S4C accepts Evidence, Journal, and checkpoint roots only as opaque identity strings. It has
no API parameter carrying their filesystem paths or stores and imports none of their writer
implementations. The only writable path is the Stage-4 adoption receipt path and its
same-directory temporary sibling.

Consequently adoption cannot:

- mint or admit Evidence;
- append the authoritative EventJournal;
- seal a training snapshot or checkpoint;
- create a Challenger;
- promote/reject a Champion;
- grant Permission;
- execute Physics/account transitions; or
- release/increment a generation.

## Verification

Read-only verification is available through:

```bash
python scripts/verify_r11_stage4_authority_adoption.py \
  --receipt <receipt.json> \
  --contract <accepted-source-contract.json>
```

The optional contract file has this shape (values are illustrative placeholders only and
must not be used as real authority):

```json
{
  "accepted_source": {
    "source_repo": "owner/repo",
    "source_sha": "<40-hex-commit>",
    "semantic_freeze_identity": "<40-hex-git-blob>",
    "source_generation": 0,
    "champion_identity": "<accepted-id>",
    "champion_hash": "<64-hex>",
    "checkpoint_identity": "<accepted-id>",
    "checkpoint_hash": "<64-hex>",
    "evidence_root_identity": "<accepted-id>",
    "journal_head_identity": "<accepted-id>",
    "checkpoint_root_identity": "<accepted-id>"
  },
  "target_r11_authority_identity": "<canonical-r11-authority-id>"
}
```

Omitting `--contract` checks only receipt self-consistency and is diagnostic; canonical
runtime adoption must supply the accepted-source contract.

## Integration boundary

S4C does not acquire runtime ownership/fencing, choose persistent roots, retire legacy
writers, or start the runtime. Final integration must call it only after independently
establishing the trusted accepted-source contract and appropriate runtime ownership. It must
then pass the validated adoption identity to the canonical lifecycle without treating the
receipt as a generation-release or scientific event.
