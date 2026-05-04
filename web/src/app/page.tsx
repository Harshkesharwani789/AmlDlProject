/* eslint-disable @next/next/no-img-element */
"use client";

import { motion, Variants } from "framer-motion";
import Link from "next/link";
import { ArrowRight, ChevronDown, ExternalLink, Activity, Target, Layers, ShieldCheck, Zap, LucideIcon } from "lucide-react";

const fadeUp: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0, 
    transition: { 
      duration: 0.5, 
      ease: [0.21, 0.45, 0.32, 0.9] as const
    } 
  },
};

const stagger: Variants = { 
  visible: { 
    transition: { 
      staggerChildren: 0.1,
      delayChildren: 0.2
    } 
  } 
};

export default function Home() {
  return (
    <main className="bg-[#020617] text-white selection:bg-purple-500/30 selection:text-purple-200">
      {/* ── STICKY NAVBAR ── */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center text-white font-bold">D</div>
            <span className="font-semibold text-white tracking-tight">DenseVision</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-zinc-400">
            <a href="#problem" className="hover:text-purple-400 transition-colors">Problem</a>
            <a href="#approach" className="hover:text-purple-400 transition-colors">Architecture</a>
            <a href="#results" className="hover:text-purple-400 transition-colors">Results</a>
            <a href="#validation" className="hover:text-purple-400 transition-colors">Validation</a>
          </div>
          <Link href="/demo" className="bg-white text-zinc-950 px-5 py-2 rounded-full text-sm font-medium hover:bg-zinc-200 transition-all hover:scale-105 active:scale-95 shadow-lg shadow-purple-500/20">
            Live Demo
          </Link>
        </div>
      </nav>

      {/* ── SLIDE 1: HERO ── */}
      <section className="slide-section relative overflow-hidden pt-32 dot-pattern">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-purple-900/20 blur-[120px] rounded-full -z-10" />
        <div className="max-w-5xl mx-auto px-6 text-center relative z-10">
          <motion.div initial="hidden" animate="visible" variants={stagger}>
            <motion.div variants={fadeUp} className="inline-flex items-center gap-2 bg-purple-500/10 border border-purple-500/20 px-3 py-1 rounded-full text-purple-400 text-xs font-semibold tracking-wide uppercase mb-8">
              <Activity className="w-3.5 h-3.5" /> Advanced Machine Learning &amp; Deep Learning
            </motion.div>
            <motion.h1 variants={fadeUp} className="font-serif text-6xl md:text-8xl lg:text-9xl leading-[0.9] text-white tracking-tight">
              High-Density<br />
              <span className="italic bg-gradient-to-r from-purple-400 to-violet-600 bg-clip-text text-transparent">Segmentation</span>
            </motion.h1>
            <motion.p variants={fadeUp} className="mt-8 text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed font-light">
              A three-phase hybrid framework: HOG+SVM baseline, YOLACT deep detector,
              and GMM+KDE spatial reasoning fusion on SKU-110K.
            </motion.p>
            <motion.div variants={fadeUp} className="mt-4 flex items-center justify-center gap-2 text-zinc-500 font-medium">
              <span>Siddhartha Shukla</span>
              <span className="text-zinc-800">&bull;</span>
              <span>Harsh Gupta</span>
            </motion.div>
            <motion.div variants={fadeUp} className="mt-12 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/demo" className="w-full sm:w-auto inline-flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-violet-700 text-white px-8 py-4 rounded-2xl font-semibold text-base hover:opacity-90 transition-all shadow-xl shadow-purple-900/40 hover:-translate-y-1">
                Try Live Demo <ArrowRight className="w-5 h-5" />
              </Link>
              <a href="#problem" className="w-full sm:w-auto inline-flex items-center justify-center gap-2 border border-white/10 bg-white/5 px-8 py-4 rounded-2xl text-base font-semibold text-white hover:bg-white/10 transition-all">
                View Presentation
              </a>
            </motion.div>
            <motion.div variants={fadeUp} className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-8 max-w-3xl mx-auto">
              {(
                [
                  ["~10M", "Parameters", Zap],
                  ["147", "Avg Obj/Image", Target],
                  ["8.3", "FPS (ONNX)", Activity],
                  ["3.097", "Best Val Loss", ShieldCheck],
                ] as [string, string, LucideIcon][]
              ).map(([v, l, Icon]) => (
                <div key={l} className="p-6 rounded-2xl border border-white/5 bg-white/5 hover:border-purple-500/50 hover:bg-purple-500/10 transition-colors group">
                  <Icon className="w-5 h-5 text-zinc-500 group-hover:text-purple-400 transition-colors mb-3 mx-auto" />
                  <p className="text-3xl font-serif text-white">{v}</p>
                  <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mt-1">{l}</p>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }} className="absolute bottom-10 left-1/2 -translate-x-1/2">
          <ChevronDown className="w-6 h-6 text-zinc-300 animate-bounce" />
        </motion.div>
      </section>

      {/* ── SLIDE 2: PROBLEM ── */}
      <section id="problem" className="slide-section border-y border-white/5 bg-zinc-950/30">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <div className="grid lg:grid-cols-2 gap-16 items-center">
              <motion.div variants={fadeUp}>
                <div className="inline-flex items-center gap-2 text-purple-400 font-semibold text-sm tracking-widest uppercase mb-4">
                  <div className="w-8 h-px bg-purple-500" /> The Challenge
                </div>
                <h2 className="font-serif text-5xl md:text-6xl text-white mb-6 tracking-tight">Why Dense Detection<br/><span className="italic text-zinc-500">is Hard</span></h2>
                <p className="text-zinc-400 text-lg leading-relaxed mb-8 font-light">
                  SKU-110K contains retail shelf images with 147 products per frame on average.
                  Standard NMS aggressively removes overlapping detections, killing recall.
                </p>
                <div className="space-y-4">
                  {[
                    ["Total images", "11,762"],
                    ["Avg objects/image", "147.4"],
                    ["Total annotations", "1.73M"],
                    ["Train / Val / Test", "8,219 / 588 / 2,936"],
                  ].map(([l, v]) => (
                    <div key={l} className="flex justify-between items-center p-4 rounded-xl border border-white/5 bg-white/5 shadow-sm hover:border-purple-500/20 transition-colors">
                      <span className="text-zinc-500 font-medium">{l}</span>
                      <span className="font-bold text-white tabular-nums">{v}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
              <motion.div variants={fadeUp} className="relative">
                <div className="absolute -inset-4 bg-purple-500/10 blur-3xl rounded-full -z-10" />
                <div className="rounded-3xl overflow-hidden border border-white/10 bg-zinc-900 shadow-2xl p-2">
                  <img src="/results/objects_per_image_histogram.png" alt="Object count distribution" className="w-full rounded-2xl opacity-80 hover:opacity-100 transition-opacity" />
                  <div className="p-4 flex items-start gap-3">
                    <Activity className="w-4 h-4 text-purple-400 mt-0.5" />
                    <p className="text-xs text-zinc-500 leading-relaxed italic">
                      Object count distribution &mdash; long tail shows images with 400+ products, creating extreme spatial complexity.
                    </p>
                  </div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── SLIDE 3: THREE-PHASE APPROACH ── */}
      <section id="approach" className="slide-section">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <motion.div variants={fadeUp} className="text-center mb-16">
              <div className="inline-flex items-center gap-2 text-purple-400 font-semibold text-sm tracking-widest uppercase mb-4">
                <div className="w-8 h-px bg-purple-500" /> Methodology
              </div>
              <h2 className="font-serif text-5xl md:text-6xl text-white mb-6">Three-Phase Architecture</h2>
              <p className="text-zinc-400 text-lg max-w-2xl mx-auto font-light">
                Progressive evolution from classical ML to a hybrid system with spatial reasoning.
              </p>
            </motion.div>

            <motion.div variants={fadeUp} className="grid md:grid-cols-3 gap-6 mb-12">
              {[
                {
                  title: "HOG + SVM",
                  phase: "Phase 1: Classical ML",
                  desc: "64x64 sliding window with 1,764-dim HOG features and linear SVM classifier.",
                  metrics: [["Precision", "86.4%", "text-zinc-300"], ["Recall", "2.1%", "text-red-400"]],
                  color: "amber"
                },
                {
                  title: "YOLACT",
                  phase: "Phase 2: Deep Learning",
                  desc: "MobileNetV3 + FPN + CBAM attention + Soft-NMS. 10M params, 20 epochs.",
                  metrics: [["Val Loss", "3.145", "text-zinc-300"], ["ONNX FPS", "8.3", "text-zinc-300"]],
                  color: "violet"
                },
                {
                  title: "YOLACT + Spatial",
                  phase: "Phase 3: Hybrid Fusion",
                  desc: "GMM row detection + KDE density prior + gated attention + confidence recalibration.",
                  metrics: [["Val Loss", "3.097", "text-zinc-300"], ["Overhead", "+37K", "text-zinc-300"]],
                  color: "purple"
                }
              ].map((item) => (
                <div key={item.title} className={`group p-8 rounded-[2rem] border transition-all hover:shadow-purple-500/10 hover:-translate-y-1 ${
                  item.color === 'amber' ? 'border-amber-900/30 bg-amber-950/20' : 
                  item.color === 'violet' ? 'border-violet-900/30 bg-violet-950/20' : 
                  'border-purple-900/30 bg-purple-950/20'
                }`}>
                  <div className={`text-[10px] font-bold uppercase tracking-[0.2em] mb-4 ${
                    item.color === 'amber' ? 'text-amber-500' : 
                    item.color === 'violet' ? 'text-violet-400' : 
                    'text-purple-400'
                  }`}>{item.phase}</div>
                  <h3 className="font-serif text-3xl text-white mb-4 tracking-tight group-hover:text-purple-400 transition-colors">{item.title}</h3>
                  <p className="text-zinc-500 text-sm leading-relaxed mb-8 font-light">
                    {item.desc}
                  </p>
                  <div className="space-y-3">
                    {item.metrics.map(([l, v, c]) => (
                      <div key={l} className="flex justify-between items-center py-2 border-b border-white/5">
                        <span className="text-[10px] uppercase font-bold text-zinc-600">{l}</span>
                        <span className={`text-sm font-bold tabular-nums ${c}`}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </motion.div>

            {/* Pipeline flow */}
            <motion.div variants={fadeUp} className="relative p-1 bg-white/5 rounded-[2.5rem] overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-violet-500/20 opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative bg-zinc-950 rounded-[2.25rem] p-8 md:p-12 border border-white/5">
                <div className="flex flex-wrap items-center justify-center gap-4">
                  {[
                    ["Input", "550x550"],
                    ["MobileNetV3", "5.4M params"],
                    ["FPN + CBAM", "Attention"],
                    ["ProtoNet", "32 masks"],
                    ["Spatial Engine", "GMM + KDE"],
                    ["Gated Fusion", "gate=0.408"],
                    ["Soft-NMS", "\u03C3=0.5"],
                    ["Output", "Detections"],
                  ].map(([title, sub], i, arr) => (
                    <div key={title} className="flex items-center gap-4">
                      <div className="bg-white/5 border border-white/5 rounded-2xl px-6 py-4 text-center hover:border-purple-500/50 hover:bg-purple-500/10 transition-all cursor-default group/item">
                        <p className="font-bold text-white text-xs tracking-tight group-hover/item:text-purple-300 transition-colors">{title}</p>
                        <p className="text-[10px] text-zinc-500 mt-1 uppercase font-semibold">{sub}</p>
                      </div>
                      {i < arr.length - 1 && <ArrowRight className="w-4 h-4 text-zinc-700 shrink-0 hidden sm:block" />}
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>

            <motion.div variants={fadeUp} className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mt-12">
              {[
                ["MobileNetV3-Large", "NAS backbone with SE blocks. 88% lighter than ResNet-101."],
                ["CBAM Attention", "Channel + spatial attention for dense scene discrimination."],
                ["Spatial Engine", "GMM row detection + KDE density prior for spatial features."],
                ["Soft-NMS", "Gaussian decay preserves valid overlapping detections (Critical)."],
              ].map(([title, desc]) => (
                <div key={title} className="p-6 rounded-2xl border border-white/5 bg-white/5 hover:border-purple-500/30 transition-colors group">
                  <h3 className="font-bold text-zinc-400 group-hover:text-purple-300 transition-colors mb-2 text-xs tracking-tight uppercase">{title}</h3>
                  <p className="text-xs text-zinc-500 leading-relaxed font-light">{desc}</p>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── SLIDE 4: RESULTS ── */}
      <section id="results" className="slide-section bg-[#020617] text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-1/2 h-full bg-purple-600/10 blur-[150px] -z-0" />
        <div className="max-w-6xl mx-auto px-6 relative z-10">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <motion.div variants={fadeUp} className="mb-16">
              <div className="inline-flex items-center gap-2 text-purple-400 font-semibold text-sm tracking-widest uppercase mb-4">
                <div className="w-8 h-px bg-purple-400" /> Performance
              </div>
              <h2 className="font-serif text-5xl md:text-6xl text-white mb-6">Evaluation Results</h2>
              <p className="text-zinc-400 text-lg max-w-2xl font-light">
                Trained on 8,219 images for 20 epochs on H100 GPU. Loss reduced 63% with consistent convergence.
              </p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-8 mb-12">
              {/* Loss convergence */}
              <motion.div variants={fadeUp} className="bg-zinc-800/50 backdrop-blur-md border border-zinc-700/50 rounded-[2.5rem] p-8 md:p-10">
                <h3 className="font-serif text-2xl text-white mb-2">YOLACT Loss Convergence</h3>
                <p className="text-xs text-zinc-500 mb-8 uppercase tracking-widest font-bold">Training progress &mdash; 20 Epochs</p>
                <div className="space-y-4">
                  {[
                    [1, 8.620], [4, 5.541], [8, 4.414], [12, 4.133],
                    [16, 3.909], [20, 3.808],
                  ].map(([ep, v]) => (
                    <div key={ep} className="flex items-center gap-4 group">
                      <span className="text-[10px] font-bold text-zinc-500 w-8">E{ep}</span>
                      <div className="flex-1 h-3 bg-zinc-800 rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          whileInView={{ width: `${((v as number)/8.620)*100}%` }}
                          transition={{ duration: 1, ease: "easeOut" }}
                          className="h-full bg-gradient-to-r from-purple-500 to-violet-400 rounded-full" 
                        />
                      </div>
                      <span className="text-xs font-bold text-zinc-400 tabular-nums">{(v as number).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-10 grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
                    <span className="block text-[10px] text-zinc-500 uppercase font-bold mb-1">Best Val Loss (DL)</span>
                    <span className="text-2xl font-serif text-purple-400">3.145</span>
                  </div>
                  <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
                    <span className="block text-[10px] text-zinc-500 uppercase font-bold mb-1">Hybrid Val Loss</span>
                    <span className="text-2xl font-serif text-violet-400">3.097</span>
                  </div>
                </div>
              </motion.div>

              {/* Benchmarks */}
              <motion.div variants={fadeUp} className="bg-zinc-800/50 backdrop-blur-md border border-zinc-700/50 rounded-[2.5rem] p-8 md:p-10">
                <h3 className="font-serif text-2xl text-white mb-2">Deployment Metrics</h3>
                <p className="text-xs text-zinc-500 mb-8 uppercase tracking-widest font-bold">Runtime & Benchmark Data</p>
                <div className="space-y-4">
                  {[
                    ["PyTorch FP32 (MPS)", "318ms / 3.1 FPS"],
                    ["ONNX FP32 (CPU)", "120ms / 8.3 FPS"],
                    ["Model size", "~38 MB"],
                    ["Total parameters", "10.0M + 37K"],
                    ["Training time (H100)", "~137 min"],
                  ].map(([l, v]) => (
                    <div key={l} className="flex justify-between items-center py-4 border-b border-white/5 group">
                      <span className="text-sm text-zinc-400 group-hover:text-zinc-300 transition-colors">{l}</span>
                      <span className="font-bold text-white tabular-nums group-hover:text-purple-400 transition-colors">{v}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            </div>

            {/* Ablation highlight */}
            <motion.div variants={fadeUp} className="bg-gradient-to-r from-red-500/10 to-transparent border border-red-500/20 rounded-[2rem] p-8 mb-12">
              <div className="flex flex-col md:flex-row items-center gap-8">
                <div className="shrink-0">
                  <div className="w-16 h-16 rounded-2xl bg-red-500/20 flex items-center justify-center text-red-500">
                    <ShieldCheck className="w-8 h-8" />
                  </div>
                </div>
                <div className="flex-1 text-center md:text-left">
                  <h3 className="font-bold text-white text-lg mb-1 uppercase tracking-tight">Ablation Study: Critical Finding</h3>
                  <p className="text-zinc-400 text-sm font-light">Gaussian score decay preserves valid overlapping detections in dense retail shelf layouts.</p>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-8">
                  {[
                    ["Full Hybrid", "2.73%", "baseline"],
                    ["DL Only", "3.03%", "+0.30"],
                    ["Hard NMS", "0.04%", "-2.68"],
                    ["Soft Impact", "98.4%", "mAP drop"],
                  ].map(([label, value, delta]) => (
                    <div key={label} className="text-center">
                      <p className="text-2xl font-serif text-white">{value}</p>
                      <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-tighter mt-1">{label}</p>
                      <p className="text-[10px] text-red-500 font-bold">{delta}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>

            {/* Detection samples + PR curve */}
            <motion.div variants={fadeUp} className="grid md:grid-cols-2 gap-8">
              <div className="group rounded-[2rem] overflow-hidden border border-zinc-700 bg-zinc-800 shadow-2xl">
                <img src="/results/detection_samples.png" alt="Detection samples" className="w-full grayscale hover:grayscale-0 transition-all duration-700 cursor-zoom-in" />
                <div className="p-4 bg-zinc-900 flex justify-between items-center">
                  <span className="text-[10px] font-bold text-zinc-500 uppercase">Sample Detections</span>
                  <span className="text-[10px] text-zinc-400 italic">Validation Images</span>
                </div>
              </div>
              <div className="group rounded-[2rem] overflow-hidden border border-zinc-700 bg-zinc-800 shadow-2xl">
                <img src="/results/precision_recall.png" alt="PR curves" className="w-full grayscale hover:grayscale-0 transition-all duration-700 cursor-zoom-in" />
                <div className="p-4 bg-zinc-900 flex justify-between items-center">
                  <span className="text-[10px] font-bold text-zinc-500 uppercase">PR Curves</span>
                  <span className="text-[10px] text-zinc-400 italic">Multiple IoU Thresholds</span>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── SLIDE 5: INTERPRETABILITY ── */}
      <section id="validation" className="slide-section relative">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <motion.div variants={fadeUp} className="text-center mb-16">
              <div className="inline-flex items-center gap-2 text-purple-400 font-semibold text-sm tracking-widest uppercase mb-4">
                <div className="w-8 h-px bg-purple-600" /> Interpretability
              </div>
              <h2 className="font-serif text-5xl md:text-6xl text-white mb-6 tracking-tight">Grad-CAM &amp; <span className="italic text-zinc-500">Robustness</span></h2>
              <p className="text-zinc-400 text-lg max-w-2xl mx-auto font-light">
                Analysis confirms the model learns product-relevant features and remains resilient under corruption.
              </p>
            </motion.div>

            <motion.div variants={fadeUp} className="rounded-[3rem] overflow-hidden border border-zinc-100 bg-white shadow-2xl p-4 mb-12">
              <img src="/results/gradcam_grid.png" alt="Grad-CAM heatmaps" className="w-full rounded-[2rem]" />
              <div className="p-8 text-center max-w-3xl mx-auto">
                <p className="text-sm text-zinc-500 leading-relaxed italic">
                  Grad-CAM heatmaps &mdash; warm colors show where the model focuses. It correctly attends to product boundaries and shelf edges, validating the efficacy of CBAM attention blocks.
                </p>
              </div>
            </motion.div>

            <div className="grid md:grid-cols-2 gap-8">
              <motion.div variants={fadeUp} className="rounded-[2.5rem] overflow-hidden border border-zinc-100 bg-white shadow-xl">
                <img src="/results/robustness_analysis.png" alt="Robustness analysis" className="w-full" />
                <div className="p-6 border-t border-zinc-50">
                  <h4 className="font-bold text-xs uppercase tracking-widest text-zinc-400 mb-2">Corruption Resilience</h4>
                  <p className="text-sm text-zinc-600 font-light">AP remains stable under noise, blur, and brightness variations.</p>
                </div>
              </motion.div>
              <motion.div variants={fadeUp} className="rounded-[2.5rem] overflow-hidden border border-zinc-100 bg-white shadow-xl">
                <img src="/results/density_analysis.png" alt="Density analysis" className="w-full" />
                <div className="p-6 border-t border-zinc-50">
                  <h4 className="font-bold text-xs uppercase tracking-widest text-zinc-400 mb-2">Density-Wise mAP</h4>
                  <p className="text-sm text-zinc-600 font-light">Performance peaks at High density (9.09%) due to spatial prior effectiveness.</p>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── SLIDE 6: CTA ── */}
      <section className="slide-section bg-gradient-to-b from-zinc-950 to-purple-950 relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(139,92,246,0.1),transparent_70%)]" />
        <div className="max-w-4xl mx-auto px-6 text-center relative z-10">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <motion.h2 variants={fadeUp} className="font-serif text-6xl md:text-8xl text-white mb-8 tracking-tighter">
              See it in <span className="italic opacity-80 underline decoration-purple-500/30 underline-offset-8">action</span>
            </motion.h2>
            <motion.p variants={fadeUp} className="text-purple-100 text-xl md:text-2xl mb-12 max-w-2xl mx-auto font-light leading-relaxed">
              Upload a retail shelf image and compare all three architectural phases in real-time.
            </motion.p>
            <motion.div variants={fadeUp}>
              <Link href="/demo" className="inline-flex items-center gap-4 bg-white text-zinc-950 px-12 py-6 rounded-full font-bold hover:bg-zinc-100 transition-all text-xl shadow-2xl hover:scale-105 active:scale-95">
                Launch Inference Engine <ArrowRight className="w-6 h-6" />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-black border-t border-white/5 py-16">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8 mb-12">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-6 h-6 bg-purple-600 rounded-md" />
                <span className="font-bold text-white tracking-tight">DenseVision</span>
              </div>
              <p className="text-zinc-500 text-sm max-w-sm font-light">
                A research project focused on high-density object segmentation in complex retail environments.
              </p>
            </div>
            <div className="flex gap-12">
              <div className="space-y-4">
                <h5 className="text-[10px] font-bold text-zinc-600 uppercase tracking-[0.2em]">Team</h5>
                <ul className="text-sm text-zinc-400 space-y-2">
                  <li>Siddhartha Shukla</li>
                  <li>Harsh Gupta</li>
                </ul>
              </div>
              <div className="space-y-4">
                <h5 className="text-[10px] font-bold text-zinc-600 uppercase tracking-[0.2em]">Links</h5>
                <ul className="text-sm text-zinc-400 space-y-2">
                  <li><a href="https://github.com/Harshkesharwani789/AmlDlProject" target="_blank" rel="noopener noreferrer" className="hover:text-purple-400 flex items-center gap-1.5 transition-colors"><ExternalLink className="w-3 h-3" /> GitHub</a></li>
                  <li><Link href="/demo" className="hover:text-purple-400 transition-colors">Demo</Link></li>
                </ul>
              </div>
            </div>
          </div>
          <div className="pt-8 border-t border-zinc-200/50 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-zinc-400 font-medium">
            <p>&copy; 2026 DenseVision Project. All rights reserved.</p>
            <p>Department of Computer Science &middot; Faculty of Engineering</p>
          </div>
        </div>
      </footer>
    </main>
  );
}
 
