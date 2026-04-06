const { test, expect } = require('@playwright/test');

test('add todo flow', async ({ page }) => {
  await page.goto('https://demo.playwright.dev/todomvc');

  // Add first todo
  await page.locator('.new-todo').fill('Buy groceries');
  await page.locator('.new-todo').press('Enter');

  // Verify it appears in the list
  await expect(page.locator('.todo-list li')).toHaveCount(1);
  await expect(page.locator('.todo-list li').first()).toContainText('Buy groceries');

  // Add second todo
  await page.locator('.new-todo').fill('Walk the dog');
  await page.locator('.new-todo').press('Enter');

  // Verify both todos are in the list
  await expect(page.locator('.todo-list li')).toHaveCount(2);
  await expect(page.locator('.todo-list li').nth(1)).toContainText('Walk the dog');
});
