/** One-time import of the pre-backend browser transcript (localStorage) into the local backend. */
import {
  type api as Api,
  type ConversationSummary,
  type ImportedConversation,
  type ManualId,
} from "./api";
import { manualIds } from "./manuals";

type LegacyRecord = Record<string, unknown>;
type MigrationState = Record<
  string,
  { conversation_ids: string[]; status: "pending" }
>;

const legacyStorageKey = "civil-reference-chats-v1";
const migrationStorageKey = "civil-reference-chat-import-v1";

function isUuid(value: unknown): value is string {
  return (
    typeof value === "string" &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
      value,
    )
  );
}

async function sha256(value: string) {
  const bytes = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return [...new Uint8Array(bytes)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function uuidFromHash(hash: string) {
  const chars = hash.slice(0, 32).split("");
  chars[12] = "5";
  chars[16] = ((Number.parseInt(chars[16], 16) & 0x3) | 0x8).toString(16);
  return `${chars.slice(0, 8).join("")}-${chars.slice(8, 12).join("")}-${chars.slice(12, 16).join("")}-${chars.slice(16, 20).join("")}-${chars.slice(20, 32).join("")}`;
}

async function normalizedId(
  rawHash: string,
  kind: string,
  index: number,
  original: unknown,
) {
  if (isUuid(original)) return original;
  return uuidFromHash(
    await sha256(
      `${rawHash}:${kind}:${index}:${typeof original === "string" ? original : ""}`,
    ),
  );
}

async function parseLegacyImport(
  raw: string,
): Promise<{ hash: string; conversations: ImportedConversation[] }> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(
      "The saved browser transcript is not valid JSON. It has not been changed.",
    );
  }
  if (!Array.isArray(parsed))
    throw new Error(
      "The saved browser transcript is not a conversation array. It has not been changed.",
    );
  const hash = await sha256(raw);
  const conversations: ImportedConversation[] = [];
  const conversationIds = new Set<string>();
  const messageIds = new Set<string>();
  for (
    let conversationIndex = 0;
    conversationIndex < parsed.length;
    conversationIndex += 1
  ) {
    const conversation = parsed[conversationIndex];
    if (
      typeof conversation !== "object" ||
      conversation === null ||
      Array.isArray(conversation)
    ) {
      throw new Error(
        "The saved browser transcript has an invalid conversation. It has not been changed.",
      );
    }
    const legacyConversation = conversation as LegacyRecord;
    if (
      typeof legacyConversation.title !== "string" ||
      !Array.isArray(legacyConversation.messages)
    ) {
      throw new Error(
        "The saved browser transcript has an invalid conversation. It has not been changed.",
      );
    }
    const createdAt = legacyConversation.createdAt;
    if (
      typeof createdAt !== "number" ||
      !Number.isSafeInteger(createdAt) ||
      !Number.isFinite(createdAt)
    ) {
      throw new Error(
        "The saved browser transcript has an invalid conversation timestamp. It has not been changed.",
      );
    }
    const id = await normalizedId(
      hash,
      "conversation",
      conversationIndex,
      legacyConversation.id,
    );
    if (conversationIds.has(id))
      throw new Error(
        "The saved browser transcript has duplicate conversation identities. It has not been changed.",
      );
    conversationIds.add(id);
    const messages: ImportedConversation["messages"] = [];
    for (
      let messageIndex = 0;
      messageIndex < legacyConversation.messages.length;
      messageIndex += 1
    ) {
      const message = legacyConversation.messages[messageIndex];
      if (
        typeof message !== "object" ||
        message === null ||
        Array.isArray(message)
      ) {
        throw new Error(
          "The saved browser transcript has an invalid message. It has not been changed.",
        );
      }
      const legacyMessage = message as LegacyRecord;
      if (
        (legacyMessage.role !== "user" && legacyMessage.role !== "assistant") ||
        typeof legacyMessage.content !== "string"
      ) {
        throw new Error(
          "The saved browser transcript has an invalid message. It has not been changed.",
        );
      }
      const messageCreatedAt = legacyMessage.createdAt;
      if (
        typeof messageCreatedAt !== "number" ||
        !Number.isSafeInteger(messageCreatedAt) ||
        !Number.isFinite(messageCreatedAt)
      ) {
        throw new Error(
          "The saved browser transcript has an invalid message timestamp. It has not been changed.",
        );
      }
      const sources = legacyMessage.sources ?? [];
      if (
        !Array.isArray(sources) ||
        sources.some(
          (source) => typeof source !== "string" || !manualIds.has(source),
        )
      ) {
        throw new Error(
          "The saved browser transcript has an invalid manual scope. It has not been changed.",
        );
      }
      const messageId = await normalizedId(
        hash,
        `message:${conversationIndex}`,
        messageIndex,
        legacyMessage.id,
      );
      if (messageIds.has(messageId))
        throw new Error(
          "The saved browser transcript has duplicate message identities. It has not been changed.",
        );
      messageIds.add(messageId);
      messages.push({
        id: messageId,
        role: legacyMessage.role,
        content: legacyMessage.content,
        created_at: new Date(messageCreatedAt).toISOString(),
        selected_manual_ids: [...sources] as ManualId[],
      });
    }
    conversations.push({
      id,
      title: legacyConversation.title,
      created_at: new Date(createdAt).toISOString(),
      messages,
    });
  }
  return { hash, conversations };
}

/**
 * Imports the saved browser transcript, then removes the legacy keys only after the
 * backend list confirms every imported conversation. Returns the refreshed list, or
 * null when there is nothing to import (or the caller cancelled after parsing).
 * The original stored value is never modified before that verified reload.
 */
export async function importLegacyConversations(
  storage: Storage,
  client: Pick<typeof Api, "importConversations" | "listConversations">,
  isCancelled: () => boolean,
): Promise<ConversationSummary[] | null> {
  const raw = storage.getItem(legacyStorageKey);
  if (!raw) return null;
  const migration = await parseLegacyImport(raw);
  if (isCancelled()) return null;
  let state: MigrationState = {};
  const savedState = storage.getItem(migrationStorageKey);
  if (savedState) {
    try {
      const parsedState: unknown = JSON.parse(savedState);
      if (
        typeof parsedState === "object" &&
        parsedState !== null &&
        !Array.isArray(parsedState)
      )
        state = parsedState as MigrationState;
    } catch {
      throw new Error(
        "The browser migration state is invalid. The saved transcript has not been changed.",
      );
    }
  }
  state[migration.hash] = {
    conversation_ids: migration.conversations.map(
      (conversation) => conversation.id,
    ),
    status: "pending",
  };
  storage.setItem(migrationStorageKey, JSON.stringify(state));
  await client.importConversations(migration.conversations);
  const refreshed = await client.listConversations();
  const known = new Set(
    refreshed.conversations.map((conversation) => conversation.id),
  );
  if (
    !migration.conversations.every((conversation) => known.has(conversation.id))
  ) {
    throw new Error(
      "The import was acknowledged but has not yet appeared in the backend. The saved transcript was retained.",
    );
  }
  storage.removeItem(legacyStorageKey);
  storage.removeItem(migrationStorageKey);
  return refreshed.conversations;
}
