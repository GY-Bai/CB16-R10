# CB16 R11 Stage-4 S4D — Runtime Singleton Lease & Fencing

## Scope

S4D provides one single-machine ownership primitive. It does not implement distributed consensus, scientific adjudication, writer rewiring, storage ownership, authority adoption, or legacy retirement.

## Safety model

A stable local lock inode is held with non-blocking POSIX `flock` for the lifetime of the authoritative owner. Persistent metadata carries a monotonically increasing `epoch` plus a random owner nonce. Successive clean acquisitions and explicit dead-owner recoveries always advance the epoch. The pair `(epoch, owner_nonce)` is the fencing token.

PID is diagnostic metadata only. There is no TTL and no wall-clock authority. If persistent state says `ACTIVE` but the kernel lock is free, ordinary acquire fails closed with `AuthorityRecoveryRequiredError`; an operator/runtime recovery path must call `recover_after_dead_owner`, which advances the fence. Corrupt, missing, symlinked, or schema-invalid metadata fails closed.

A copied token is insufficient to assert authority: `assert_fencing_token` requires both an exact durable token match and a live lock owned by the calling process. The at-fork hook closes inherited lock descriptors in children so a forked worker cannot accidentally keep the parent's ownership alive after parent death.

Release writes durable `RELEASED` state before unlocking. A release with the wrong fence cannot unlock or authorize an old writer.

## Integration contract

Final Stage-4 integration must require a current `FencingTokenR11` at every authoritative mutation boundary and call the lease provider's assertion immediately before mutation. At minimum this applies to authoritative journal append, Evidence admission, snapshot seal, Challenger/Champion transitions, checkpoint seal, generation release, Permission grant, and Physics/account execution authority. S4D intentionally does not modify those existing writers.

The canonical persistent root and filesystem placement are owned by later state-root/integration work. S4D assumes a single host and local filesystem locking semantics; it is not valid as a cross-machine coordination mechanism.

## Hostile qualification

The test suite covers simultaneous process acquisition, second-owner denial, same-process double acquire, stale fencing after clean handoff, SIGKILL and explicit recovery, stale fencing after crash recovery, PID-reuse simulation, corrupt/missing metadata, forged release fencing, live-owner recovery denial, and symlink metadata rejection.

## Scientific boundary

This primitive reads/writes only Stage-4 control metadata in disposable/runtime control roots. It creates no Evidence, generation, Champion/Challenger decision, Permission, market data, or scientific verdict and changes no frozen semantic contract.
