/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Upload,
  ArrowLeft,
  Loader2,
  ImageIcon,
  X,
  Maximize2,
  Download,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Brain,
  ScanLine,
  SlidersHorizontal,
  Target,
  Zap,
  Activity,
  ShieldCheck,
  LucideIcon,
} from "lucide-react";

type ModelType = "yolact" | "hogsvm" | "hybrid";

interface Detection {
  box: [number, number, number, number];
  score: number;
  label: number;
}

interface InferenceResult {
  detections: Detection[];
  num_detections: number;
  inference_time_ms: number;
  image_width: number;
  image_height: number;
  model?: string;
}

const MODEL_INFO: Record<ModelType, { name: string; tag: string; desc: string; endpoint: string; color: string }> = {
  yolact: {
    name: "YOLACT",
    tag: "Deep Learning",
    desc: "MobileNetV3 + FPN + CBAM + Soft-NMS",
    endpoint: "/api/inference",
    color: "violet",
  },
  hybrid: {
    name: "Hybrid",
    tag: "ML + DL Fusion",
    desc: "YOLACT + GMM + KDE Spatial Reasoning",
    endpoint: "/api/inference-hybrid",
    color: "purple",
  },
  hogsvm: {
    name: "HOG + SVM",
    tag: "Classical ML",
    desc: "HOG features + Linear SVM + Sliding Window",
    endpoint: "/api/inference-baseline",
    color: "amber",
  },
};

const SAMPLE_IMAGES = [
  { src: "/samples/shelf_dense.png", label: "Dense Shelf" },
  { src: "/samples/shelf_medium.png", label: "Medium Shelf" },
  { src: "/samples/shelf_sparse.png", label: "Sparse Shelf" },
];

export default function DemoPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelType>("yolact");
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const [fullscreen, setFullscreen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });
  const panOffset = useRef({ x: 0, y: 0 });
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Filter detections by confidence threshold (client-side, no re-inference)
  const filteredDetections = useMemo(() => {
    if (!result) return [];
    return result.detections.filter((d) => d.score >= confidenceThreshold);
  }, [result, confidenceThreshold]);

  const drawDetections = useCallback((detections: Detection[]) => {
    if (!preview) return;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.box;
        const w = x2 - x1;
        const h = y2 - y1;

        let r: number, g: number, b: number;
        if (det.score > 0.5) {
          const t = (det.score - 0.5) / 0.5;
          r = 139; g = 92; b = 246; // Violet-500
        } else {
          r = 82; g = 82; b = 91; // Zinc-600
        }

        const lineW = Math.max(2, Math.min(6, img.width / 150));

        ctx.shadowColor = `rgba(0, 0, 0, 0.3)`;
        ctx.shadowBlur = 4;
        ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.lineWidth = lineW;
        ctx.strokeRect(x1, y1, w, h);
        ctx.shadowBlur = 0;

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.1)`;
        ctx.fillRect(x1, y1, w, h);

        if (det.score > 0.1) {
          const label = `${(det.score * 100).toFixed(0)}%`;
          const fontSize = Math.max(12, img.width / 40);
          ctx.font = `bold ${fontSize}px sans-serif`;
          const metrics = ctx.measureText(label);
          const labelH = fontSize + 6;
          const labelW = metrics.width + 10;

          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.fillRect(x1 - lineW/2, y1 - labelH, labelW, labelH);

          ctx.fillStyle = "#fff";
          ctx.fillText(label, x1 + 5, y1 - 6);
        }
      });

      setResultImage(canvas.toDataURL("image/png"));
    };
    img.src = preview;
  }, [preview]);

  // Redraw detections when threshold changes
  useEffect(() => {
    if (result && preview) {
      drawDetections(filteredDetections);
    }
  }, [confidenceThreshold, result, preview, drawDetections, filteredDetections]);

  const closeFullscreen = useCallback(() => {
    setFullscreen(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && fullscreen) closeFullscreen();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [fullscreen, closeFullscreen]);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file");
      return;
    }
    setFile(f);
    setResult(null);
    setResultImage(null);
    setError(null);
    setConfidenceThreshold(0.25);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const handleSampleImage = async (src: string) => {
    setResult(null);
    setResultImage(null);
    setError(null);
    setConfidenceThreshold(0.25);
    setPreview(src);

    // Convert to File for the API
    const res = await fetch(src);
    const blob = await res.blob();
    const f = new File([blob], src.split("/").pop() || "sample.png", { type: blob.type });
    setFile(f);
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const runInference = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setResultImage(null);

    const modelConfig = MODEL_INFO[selectedModel];

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch(modelConfig.endpoint, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Inference failed (${res.status})`);
      }

      const data: InferenceResult = await res.json();
      data.model = modelConfig.name;
      setResult(data);
      const filtered = data.detections.filter((d) => d.score >= confidenceThreshold);
      drawDetections(filtered);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setResultImage(null);
    setError(null);
    setConfidenceThreshold(0.25);
  };

  const openFullscreen = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setFullscreen(true);
  };

  const handleZoomIn = () => setZoom((z) => Math.min(z * 1.3, 8));
  const handleZoomOut = () => setZoom((z) => Math.max(z / 1.3, 0.5));
  const handleZoomReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoom((z) => Math.min(z * 1.1, 8));
    } else {
      setZoom((z) => Math.max(z / 1.1, 0.5));
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom <= 1) return;
    setIsPanning(true);
    panStart.current = { x: e.clientX, y: e.clientY };
    panOffset.current = { x: pan.x, y: pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    setPan({
      x: panOffset.current.x + (e.clientX - panStart.current.x),
      y: panOffset.current.y + (e.clientY - panStart.current.y),
    });
  };

  const handleMouseUp = () => setIsPanning(false);

  const downloadImage = () => {
    if (!resultImage) return;
    const link = document.createElement("a");
    link.download = `detection_${selectedModel}_${Date.now()}.png`;
    link.href = resultImage;
    link.click();
  };

  const modelInfo = MODEL_INFO[selectedModel];

  return (
    <main className="min-h-screen bg-[#020617] text-white dot-pattern">
      {/* ── STICKY NAVBAR ── */}
      <nav className="sticky top-0 z-50 glass border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" className="p-2 hover:bg-white/5 rounded-full transition-colors">
              <ArrowLeft className="w-5 h-5 text-zinc-400" />
            </Link>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center text-white font-bold">D</div>
              <span className="font-semibold text-white tracking-tight hidden sm:block">DenseVision Inference Engine</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {result && (
              <button onClick={reset} className="text-sm font-medium text-zinc-400 hover:text-white transition-colors">
                Reset
              </button>
            )}
            <div className="w-px h-6 bg-white/10" />
            <a href="https://github.com/Harshkesharwani789/AmlDlProject" target="_blank" rel="noopener noreferrer" className="p-2 hover:bg-white/5 rounded-full transition-colors text-zinc-400">
              <Activity className="w-5 h-5" />
            </a>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-12 gap-8">
          {/* ── LEFT: CONTROLS & SELECTION (4 cols) ── */}
          <div className="lg:col-span-4 space-y-6">
            {/* Model Selector Card */}
            <div className="bg-white/5 rounded-[2rem] border border-white/5 p-6 shadow-sm backdrop-blur-sm">
              <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500 mb-6">1. Select Architecture</h3>
              <div className="space-y-3">
                {(Object.keys(MODEL_INFO) as ModelType[]).map((m) => {
                  const info = MODEL_INFO[m];
                  const active = selectedModel === m;
                  const Icon = (m === 'yolact' ? Brain : m === 'hybrid' ? SlidersHorizontal : ScanLine) as LucideIcon;
                  return (
                    <button
                      key={m}
                      onClick={() => { setSelectedModel(m); setResult(null); setResultImage(null); }}
                      className={`w-full flex items-center gap-4 p-4 rounded-2xl border transition-all text-left group ${
                        active 
                          ? 'bg-purple-600/10 border-purple-500/50 shadow-sm' 
                          : 'bg-white/5 border-white/5 hover:border-white/10'
                      }`}
                    >
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors ${
                        active ? 'bg-purple-600 text-white' : 'bg-white/5 text-zinc-600 group-hover:bg-white/10'
                      }`}>
                        <Icon className="w-6 h-6" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className={`font-bold text-sm ${active ? 'text-white' : 'text-zinc-500'}`}>{info.name}</span>
                          <span className={`text-[10px] font-bold uppercase tracking-wider ${
                            active ? (m === 'yolact' ? 'text-violet-400' : m === 'hybrid' ? 'text-purple-400' : 'text-amber-500') : 'text-zinc-600'
                          }`}>{info.tag}</span>
                        </div>
                        <p className="text-[11px] text-zinc-500 mt-0.5 line-clamp-1">{info.desc}</p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Confidence Slider Card */}
            {result && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 rounded-[2rem] border border-white/5 p-6 shadow-sm backdrop-blur-sm"
              >
                <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500 mb-6">2. Confidence Filtering</h3>
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-bold text-white">Minimum Score</span>
                    <span className="bg-purple-600 text-white text-xs font-bold px-3 py-1 rounded-full">
                      {(confidenceThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.95"
                    step="0.01"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white/5 rounded-2xl p-4 border border-white/5">
                      <span className="block text-[10px] text-zinc-500 uppercase font-bold mb-1">Showing</span>
                      <span className="text-xl font-serif text-white">{filteredDetections.length}</span>
                      <span className="text-[10px] text-zinc-500 ml-1">objects</span>
                    </div>
                    <div className="bg-white/5 rounded-2xl p-4 border border-white/5">
                      <span className="block text-[10px] text-zinc-500 uppercase font-bold mb-1">Latency</span>
                      <span className="text-xl font-serif text-white">{result.inference_time_ms.toFixed(0)}</span>
                      <span className="text-[10px] text-zinc-500 ml-1">ms</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Sample Images Card */}
            {!result && (
              <div className="bg-white/5 rounded-[2rem] border border-white/5 p-6 shadow-sm backdrop-blur-sm">
                <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500 mb-6">Or use Sample</h3>
                <div className="grid grid-cols-3 gap-3">
                  {SAMPLE_IMAGES.map((sample) => (
                    <button
                      key={sample.src}
                      onClick={() => handleSampleImage(sample.src)}
                      className="group relative rounded-xl overflow-hidden border border-white/5 hover:border-purple-500/50 transition-all aspect-square"
                    >
                      <img src={sample.src} alt={sample.label} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500 opacity-60 group-hover:opacity-100" />
                      <div className="absolute inset-0 bg-purple-900/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <span className="text-[10px] text-white font-bold uppercase">Load</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* ── RIGHT: MAIN CANVAS (8 cols) ── */}
          <div className="lg:col-span-8 space-y-6">
            <div className="relative rounded-[2.5rem] overflow-hidden border border-white/5 bg-zinc-950 shadow-2xl min-h-[500px] flex items-center justify-center group">
              <AnimatePresence mode="wait">
                {!preview ? (
                  <motion.div
                    key="dropzone"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    className={`absolute inset-4 rounded-[2rem] border-2 border-dashed flex flex-col items-center justify-center gap-6 cursor-pointer transition-all ${
                      dragOver ? 'border-purple-500 bg-purple-500/10' : 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20'
                    }`}
                  >
                    <div className="w-20 h-20 rounded-[2rem] bg-zinc-900 border border-white/5 shadow-xl flex items-center justify-center text-purple-500">
                      <Upload className="w-8 h-8" />
                    </div>
                    <div className="text-center">
                      <h4 className="text-lg font-bold text-white">Drop your image here</h4>
                      <p className="text-sm text-zinc-500 mt-1">or click to browse from files</p>
                    </div>
                    <div className="flex gap-4 mt-4">
                      <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-600 uppercase tracking-widest px-4 py-2 bg-white/5 border border-white/5 rounded-full">
                        <ShieldCheck className="w-3 h-3" /> Secure
                      </div>
                      <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-600 uppercase tracking-widest px-4 py-2 bg-white/5 border border-white/5 rounded-full">
                        <Zap className="w-3 h-3" /> Fast
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="preview"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="relative w-full h-full flex items-center justify-center"
                  >
                    {/* Background Glow */}
                    <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/5 pointer-events-none" />
                    
                    <img
                      src={resultImage || preview}
                      alt="Detection preview"
                      className="max-w-full max-h-[75vh] object-contain shadow-2xl transition-all duration-500"
                    />

                    {/* Inference Overlay */}
                    {loading && (
                      <div className="absolute inset-0 bg-zinc-950/60 backdrop-blur-sm flex flex-col items-center justify-center z-20">
                        <div className="relative">
                          <Loader2 className="w-12 h-12 text-purple-500 animate-spin" />
                          <div className="absolute inset-0 blur-xl bg-purple-500/20 animate-pulse" />
                        </div>
                        <p className="mt-6 text-sm font-bold text-white uppercase tracking-[0.2em]">Inference in Progress...</p>
                        <p className="text-xs text-zinc-500 mt-2">Computing {modelInfo.name} on remote backend</p>
                      </div>
                    )}

                    {/* Success Badges */}
                    {result && !loading && (
                      <div className="absolute top-6 left-6 flex flex-col gap-2 pointer-events-none">
                        <div className="bg-purple-600 text-white text-[10px] font-bold uppercase tracking-[0.2em] px-4 py-2 rounded-xl flex items-center gap-2 shadow-xl">
                          <Activity className="w-3.5 h-3.5 text-purple-200" /> Processed
                        </div>
                        <div className="bg-zinc-900/90 backdrop-blur border border-white/10 text-white text-[10px] font-bold px-4 py-2 rounded-xl shadow-lg">
                          {result.image_width}x{result.image_height} PX
                        </div>
                      </div>
                    )}

                    {/* Floating Actions */}
                    {preview && !loading && (
                      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-zinc-900/90 backdrop-blur p-2 rounded-2xl border border-white/10 shadow-2xl transition-all group-hover:scale-105 z-30">
                        {!result ? (
                          <button
                            onClick={runInference}
                            className="flex items-center gap-3 bg-purple-600 text-white px-8 py-3 rounded-xl text-sm font-bold hover:bg-purple-700 transition-all shadow-lg shadow-purple-500/20 active:scale-95"
                          >
                            <Zap className="w-4 h-4 fill-current" /> Run Inference
                          </button>
                        ) : (
                          <>
                            <div className="w-px h-8 bg-zinc-700 mx-1" />
                            <button onClick={reset} className="p-3 hover:bg-red-900/20 rounded-xl transition-colors text-red-400" title="Remove">
                              <X className="w-5 h-5" />
                            </button>
                          </>
                        )}
                      </div>
                    )}

                    {result && !loading && (
                      <>
                        <button 
                          onClick={openFullscreen}
                          className="absolute bottom-6 right-6 p-4 bg-zinc-900/90 backdrop-blur border border-white/10 rounded-2xl text-white opacity-0 group-hover:opacity-100 transition-all hover:bg-purple-600 hover:border-purple-500 shadow-2xl z-30"
                        >
                          <Maximize2 className="w-5 h-5" />
                        </button>
                        <button 
                          onClick={downloadImage}
                          className="absolute bottom-6 right-24 p-4 bg-zinc-900/90 backdrop-blur border border-white/10 rounded-2xl text-white opacity-0 group-hover:opacity-100 transition-all hover:bg-purple-600 hover:border-purple-500 shadow-2xl z-30"
                        >
                          <Download className="w-5 h-5" />
                        </button>
                      </>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Error Message */}
            {error && (
              <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="p-4 bg-red-950/20 border border-red-900/30 rounded-2xl flex items-center gap-3 text-red-400 text-sm font-medium">
                <div className="w-8 h-8 rounded-full bg-red-900/20 flex items-center justify-center shrink-0">
                  <Activity className="w-4 h-4" />
                </div>
                {error}
              </motion.div>
            )}

            {/* Detailed Table (Visible only after inference) */}
            {result && !loading && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-white/5 rounded-[2.5rem] border border-white/5 overflow-hidden shadow-sm backdrop-blur-sm">
                <div className="p-6 border-b border-white/5 bg-white/5 flex items-center justify-between">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <Target className="w-4 h-4 text-purple-400" /> Object Inventory
                  </h3>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">{filteredDetections.length} detections matched</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="bg-white/5 text-zinc-500 uppercase tracking-wider font-bold">
                        <th className="px-6 py-4">Index</th>
                        <th className="px-6 py-4">Bounding Box [X1, Y1, X2, Y2]</th>
                        <th className="px-6 py-4 text-right">Confidence</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {filteredDetections.slice(0, 10).map((det, i) => (
                        <tr key={i} className="hover:bg-purple-500/10 transition-colors">
                          <td className="px-6 py-4 font-bold text-zinc-600">#{(i + 1).toString().padStart(2, '0')}</td>
                          <td className="px-6 py-4 font-mono text-zinc-500">[{det.box.map(Math.round).join(", ")}]</td>
                          <td className="px-6 py-4 text-right">
                            <span className={`inline-block px-3 py-1 rounded-full font-bold ${
                              det.score > 0.6 ? 'bg-purple-500/20 text-purple-300' : 'bg-white/5 text-zinc-600'
                            }`}>{(det.score * 100).toFixed(1)}%</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {filteredDetections.length > 10 && (
                  <div className="p-4 bg-white/5 text-center border-t border-white/5 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                    Showing top 10 of {filteredDetections.length} results
                  </div>
                )}
              </motion.div>
            )}
          </div>
        </div>
      </div>

      <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />

      {/* ── FULLSCREEN OVERLAY ── */}
      <AnimatePresence>
        {fullscreen && resultImage && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-zinc-950 flex flex-col"
          >
            <div className="h-16 px-6 border-b border-white/10 flex items-center justify-between text-white">
              <div className="flex items-center gap-4">
                <span className="text-sm font-bold uppercase tracking-widest text-zinc-400">Inspection Mode</span>
                <div className="w-px h-4 bg-white/20" />
                <span className="text-sm font-serif italic text-purple-400">{modelInfo.name}</span>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={handleZoomOut} className="p-2 hover:bg-white/10 rounded-lg transition-colors"><ZoomOut className="w-5 h-5" /></button>
                <span className="text-xs font-bold w-12 text-center">{(zoom * 100).toFixed(0)}%</span>
                <button onClick={handleZoomIn} className="p-2 hover:bg-white/10 rounded-lg transition-colors"><ZoomIn className="w-5 h-5" /></button>
                <div className="w-px h-4 bg-white/20 mx-2" />
                <button onClick={handleZoomReset} className="p-2 hover:bg-white/10 rounded-lg transition-colors"><RotateCcw className="w-5 h-5" /></button>
                <button onClick={closeFullscreen} className="ml-4 p-2 bg-white text-black rounded-lg hover:bg-zinc-200 transition-colors"><X className="w-5 h-5" /></button>
              </div>
            </div>
            
            <div 
              className="flex-1 relative overflow-hidden cursor-move"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onWheel={handleWheel}
            >
              <motion.div
                className="absolute inset-0 flex items-center justify-center pointer-events-none"
                animate={{
                  scale: zoom,
                  x: pan.x,
                  y: pan.y,
                }}
                transition={{ type: "spring", damping: 25, stiffness: 200, mass: 0.5 }}
              >
                <img src={resultImage} alt="Fullscreen result" className="max-w-none shadow-2xl" style={{ maxHeight: '90vh' }} />
              </motion.div>
            </div>

            <div className="h-12 bg-black/50 border-t border-white/10 px-6 flex items-center justify-center text-[10px] font-bold text-zinc-500 uppercase tracking-[0.3em]">
              Drag to pan &bull; Scroll to zoom &bull; ESC to exit
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}
