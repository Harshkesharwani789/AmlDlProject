import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";
import { runPythonInference, cleanupFile, ProfessionalResponse } from "@/lib/inference";
import { BACKEND_CONFIG } from "@/lib/constants";
import os from "os";

export async function POST(request: NextRequest) {
  const requestId = `hybrid_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  const requestStartTime = Date.now();
  
  console.log(`[${requestId}] Request received`);
  let filepath = "";

  try {
    const formData = await request.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No image provided", requestId }, { status: 400 });
    }

    // Validate file type
    if (!BACKEND_CONFIG.ALLOWED_MIME_TYPES.includes(file.type)) {
      return NextResponse.json({ 
        error: `Invalid file type: ${file.type}. Allowed types: ${BACKEND_CONFIG.ALLOWED_MIME_TYPES.join(", ")}`, 
        requestId 
      }, { status: 400 });
    }

    const uploadStartTime = Date.now();
    if (!existsSync(BACKEND_CONFIG.UPLOAD_DIR)) {
      await mkdir(BACKEND_CONFIG.UPLOAD_DIR, { recursive: true });
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filename = `upload_${requestId}.jpg`;
    filepath = join(BACKEND_CONFIG.UPLOAD_DIR, filename);
    await writeFile(filepath, buffer);
    const uploadDuration = Date.now() - uploadStartTime;

    const INFERENCE_SCRIPT = join(BACKEND_CONFIG.PROJECT_ROOT, "scripts", "inference_hybrid.py");

    if (!existsSync(INFERENCE_SCRIPT)) {
      throw new Error(`Inference script not found at ${INFERENCE_SCRIPT}`);
    }

    const processingStartTime = Date.now();
    const result = await runPythonInference(
      requestId, 
      filepath, 
      INFERENCE_SCRIPT, 
      BACKEND_CONFIG.VENV_PYTHON, 
      BACKEND_CONFIG.PROJECT_ROOT,
      BACKEND_CONFIG.TIMEOUTS.HYBRID
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
    return NextResponse.json({ 
      error: message, 
      requestId,
      type: "INFERENCE_ERROR"
    }, { status: 500 });
  } finally {
    if (filepath) {
      await cleanupFile(filepath);
    }
  }
}
