import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";
import { runPythonInference, cleanupFile, ProfessionalResponse } from "@/lib/inference";
import os from "os";

const UPLOAD_DIR = join(process.cwd(), "tmp");

export async function POST(request: NextRequest) {
  const requestId = `dl_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  const requestStartTime = Date.now();
  
  console.log(`[${requestId}] Request received`);
  let filepath = "";

  try {
    const formData = await request.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No image provided", requestId }, { status: 400 });
    }

    if (!file.type.startsWith("image/")) {
      return NextResponse.json({ error: "Invalid file type. Please upload an image.", requestId }, { status: 400 });
    }

    const uploadStartTime = Date.now();
    if (!existsSync(UPLOAD_DIR)) {
      await mkdir(UPLOAD_DIR, { recursive: true });
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filename = `upload_${requestId}.jpg`;
    filepath = join(UPLOAD_DIR, filename);
    await writeFile(filepath, buffer);
    const uploadDuration = Date.now() - uploadStartTime;

    const PROJECT_ROOT = join(process.cwd(), "..");
    const INFERENCE_SCRIPT = join(PROJECT_ROOT, "scripts", "inference_api.py");
    const VENV_PYTHON = join(PROJECT_ROOT, ".venv", "bin", "python");

    if (!existsSync(INFERENCE_SCRIPT)) {
      throw new Error(`Inference script not found at ${INFERENCE_SCRIPT}`);
    }

    const processingStartTime = Date.now();
    const result = await runPythonInference(
      requestId, 
      filepath, 
      INFERENCE_SCRIPT, 
      VENV_PYTHON, 
      PROJECT_ROOT,
      120000 // 2 minute timeout for YOLACT
    );
    const processingDuration = Date.now() - processingStartTime;

    const totalDuration = Date.now() - requestStartTime;

    const response: ProfessionalResponse = {
      ...result,
      latency: {
        upload: uploadDuration,
        processing: processingDuration,
        total: totalDuration
      },
      requestId,
      systemInfo: {
        platform: os.platform(),
        nodeVersion: process.version
      }
    };

    console.log(`[${requestId}] Success: ${result.num_detections} detections, ${totalDuration}ms`);
    return NextResponse.json(response);

  } catch (err) {
    const message = err instanceof Error ? err.message : "Inference failed";
    console.error(`[${requestId}] ERROR: ${message}`);
    return NextResponse.json({ error: message, requestId }, { status: 500 });
  } finally {
    if (filepath) {
      await cleanupFile(filepath);
    }
  }
}
