import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"

const baseStyles = readFileSync("quartz/styles/base.scss", "utf8")
const listPageStyles = readFileSync("quartz/components/styles/listPage.scss", "utf8")

test("tag links keep the full chip label together", () => {
  const tagLinkRule = /a\.internal\.tag-link\s*\{[\s\S]*?\}/.exec(baseStyles)?.[0] ?? ""

  assert.match(tagLinkRule, /white-space:\s*nowrap/)
  assert.match(tagLinkRule, /overflow-wrap:\s*normal/)
  assert.match(tagLinkRule, /word-break:\s*normal/)
})

test("page lists place wrapped tag chips below the title", () => {
  assert.match(listPageStyles, /grid-template-columns:\s*fit-content\(8em\)\s+minmax\(0,\s*1fr\)/)
  assert.doesNotMatch(listPageStyles, /grid-template-columns:[^;]*minmax\(12rem,\s*35%\)/)
  assert.match(listPageStyles, /& > \.tags\s*\{[\s\S]*?min-width:\s*0/)
  assert.match(listPageStyles, /& > \.tags\s*\{[\s\S]*?grid-column:\s*2/)
  assert.match(listPageStyles, /& > \.tags\s*\{[\s\S]*?justify-content:\s*flex-start/)
  assert.match(listPageStyles, /& > \.tags\s*\{[\s\S]*?flex-wrap:\s*wrap/)
})
