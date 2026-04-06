# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: tests/todo.spec.js >> add todo flow
- Location: tests/todo.spec.js:3:1

# Error details

```
Error: page.goto: net::ERR_NAME_NOT_RESOLVED at https://demo.playwright.dev/todomvc
Call log:
  - navigating to "https://demo.playwright.dev/todomvc", waiting until "load"

```

# Test source

```ts
  1  | const { test, expect } = require('@playwright/test');
  2  | 
  3  | test('add todo flow', async ({ page }) => {
> 4  |   await page.goto('https://demo.playwright.dev/todomvc');
     |              ^ Error: page.goto: net::ERR_NAME_NOT_RESOLVED at https://demo.playwright.dev/todomvc
  5  | 
  6  |   // Add first todo
  7  |   await page.locator('.new-todo').fill('Buy groceries');
  8  |   await page.locator('.new-todo').press('Enter');
  9  | 
  10 |   // Verify it appears in the list
  11 |   await expect(page.locator('.todo-list li')).toHaveCount(1);
  12 |   await expect(page.locator('.todo-list li').first()).toContainText('Buy groceries');
  13 | 
  14 |   // Add second todo
  15 |   await page.locator('.new-todo').fill('Walk the dog');
  16 |   await page.locator('.new-todo').press('Enter');
  17 | 
  18 |   // Verify both todos are in the list
  19 |   await expect(page.locator('.todo-list li')).toHaveCount(2);
  20 |   await expect(page.locator('.todo-list li').nth(1)).toContainText('Walk the dog');
  21 | });
  22 | 
```