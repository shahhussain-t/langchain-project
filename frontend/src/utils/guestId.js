// Utility functions for managing guest user IDs

/**
 * Generate a unique guest ID using UUID v4 format
 * @returns {string} A unique guest ID
 */
export const generateGuestId = () => {
  // Simple UUID v4 generator
  return 'guest-' + 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : ((r & 0x3) | 0x8);
    return v.toString(16);
  });
};

/**
 * Get the current guest ID from localStorage
 * @returns {string|null} The stored guest ID or null if not found
 */
export const getGuestId = () => {
  try {
    return localStorage.getItem('ai_assistant_guest_id');
  } catch (error) {
    console.error('Error reading guest ID from localStorage:', error);
    return null;
  }
};

/**
 * Store the guest ID in localStorage
 * @param {string} guestId - The guest ID to store
 */
export const setGuestId = (guestId) => {
  try {
    localStorage.setItem('ai_assistant_guest_id', guestId);
  } catch (error) {
    console.error('Error storing guest ID in localStorage:', error);
  }
};

/**
 * Generate and store a new guest ID
 * @returns {string} The newly generated guest ID
 */
export const generateAndStoreGuestId = () => {
  const newGuestId = generateGuestId();
  setGuestId(newGuestId);
  return newGuestId;
};

/**
 * Clear the stored guest ID (useful for testing or reset)
 */
export const clearGuestId = () => {
  try {
    localStorage.removeItem('ai_assistant_guest_id');
  } catch (error) {
    console.error('Error clearing guest ID from localStorage:', error);
  }
};

/**
 * Get or generate a guest ID
 * If one exists in localStorage, return it; otherwise generate a new one
 * @returns {string} A guest ID
 */
export const getOrGenerateGuestId = () => {
  let guestId = getGuestId();
  if (!guestId) {
    guestId = generateAndStoreGuestId();
  }
  return guestId;
};