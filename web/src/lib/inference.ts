import { spawn } from "child_process";
import { existsSync } from "fs";
import { unlink } from "fs/promises";
import Logger from "./logger";

// Simple Semaphore to limit concurrent Python processes
class Semaphore {
  private count: number;
  private queue: Array<() => void> = [];

  constructor(max: number) {
    this.count = max;
  }

  async acquire() {
    if (this.count > 0) {
      this.count--;
      return;
    }
    await new Promise<void>((resolve) => this.queue.push(resolve));
  }

  release() {
    this.count++;
    if (this.queue.length > 0) {
      const next = this.queue.shift();
      if (next) {
        this.count--;
        next();
      }
    }
  }
}

// Limit to 2 concurrent processes to save memory on typical small servers
const inferenceLimiter = new Semaphore(2);

export interface InferenceResult {
  detections: Array<{
    box: [number, number, number, number];
    score: number;
    label: number;
  }>;
  num_detections: number;
  inference_time_ms: number;
  image_width: number;
  image_height: number;
  metadata?: Record<string, unknown>;
}

export interface LatencyBreakdown {
  upload: number;
  processing: number;
  total: number;
}

export interface ProfessionalResponse extends InferenceResult {
  latency: LatencyBreakdown;
  requestId: string;
  systemInfo: {
    platform: string;
    nodeVersion: string;
  };
}

export async function runPythonInference(
  requestId: string,
  imagePath: string,
  scriptPath: string,
  pythonPath: string,
  cwd: string,
  timeoutMs: number = 120000
): Promise<InferenceResult> {
  await inferenceLimiter.acquire();
  Logger.info(`Semaphore acquired for ${requestId}`, { queueLength: inferenceLimiter['queue'].length });

  return new Promise((resolve, reject) => {
    const python = existsSync(pythonPath) ? pythonPath : "python3";
    
    Logger.info(`Spawning Python process for ${requestId}`, { python, scriptPath });
    const proc = spawn(python, [scriptPath, imagePath], {
      cwd: cwd,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdout = "";
    let stderr = "";

    const onExit = (error?: Error) => {
      clearTimeout(timeout);
      inferenceLimiter.release();
      if (error) {
        Logger.error(`Inference failed for ${requestId}`, { error: error.message });
        reject(error);
      }
    };

    const timeout = setTimeout(() => {
      Logger.error(`TIMEOUT for ${requestId} - killing process after ${timeoutMs}ms`);
      proc.kill();
      onExit(new Error(`Inference timed out after ${timeoutMs / 1000}s`));
    }, timeoutMs);

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      const chunk = data.toString();
      stderr += chunk;
      Logger.debug(`[python] ${chunk.trimEnd()}`, { requestId });
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        onExit(new Error(stderr || `Process exited with code ${code}`));
        return;
      }
      try {
        const result = JSON.parse(stdout);
        Logger.info(`Inference successful for ${requestId}`, { detections: result.num_detections });
        onExit();
        resolve(result);
      } catch {
        Logger.error(`Failed to parse stdout for ${requestId}`, { stdoutSnippet: stdout.slice(0, 200) });
        onExit(new Error("Failed to parse inference output (invalid JSON)"));
      }
    });

    proc.on("error", (err) => {
      onExit(new Error(`Failed to start Python: ${err.message}`));
    });
  });
}

export async function cleanupFile(path: string) {
  try {
    if (existsSync(path)) {
      await unlink(path);
      Logger.debug(`Cleaned up temporary file: ${path}`);
    }
  } catch (err) {
    Logger.error(`Failed to cleanup file ${path}`, { error: err instanceof Error ? err.message : String(err) });
  }
}
