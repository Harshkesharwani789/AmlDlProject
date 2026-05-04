import { join } from "path";

export const BACKEND_CONFIG = {
  // Directory where images are temporarily saved
  UPLOAD_DIR: join(process.cwd(), "tmp"),
  
  // Project root (one level up from 'web/')
  PROJECT_ROOT: join(process.cwd(), ".."),
  
  // Python Environment
  VENV_PYTHON: join(process.cwd(), "..", ".venv", "bin", "python"),
  
  // Timeouts in Milliseconds
  TIMEOUTS: {
    BASELINE: 60000,
    YOLACT: 120000,
    HYBRID: 180000,
  },

  // Max concurrent inference processes
  MAX_CONCURRENT_INFERENCE: 2,

  // Allowed Image Mime Types
  ALLOWED_MIME_TYPES: ["image/jpeg", "image/png", "image/webp"],
};
