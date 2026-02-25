import { useState, useRef, useCallback } from "react";
import { Upload, Sparkles, Download, X, FileAudio, Play, Pause } from "lucide-react";
import { Button } from "@/components/ui/button";
import WaveformVisualizer from "@/components/WaveformVisualizer";
import FeaturesSection from "@/components/FeaturesSection";
import StatsSection from "@/components/StatsSection";
import HowItWorks from "@/components/HowItWorks";
import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";
import heroWave from "@/assets/hero-wave.png";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";

type AppState = "idle" | "uploaded" | "enhancing" | "done" | "error";

const API_BASE = "/api";

const Index = () => {
  const [state, setState] = useState<AppState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [progress, setProgress] = useState(0);
  const [enhancedBlob, setEnhancedBlob] = useState<Blob | null>(null);
  const [enhancedUrl, setEnhancedUrl] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<string>("");
  const [rtf, setRtf] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);
  const originalAudioRef = useRef<HTMLAudioElement>(null);
  const enhancedAudioRef = useRef<HTMLAudioElement>(null);
  const [playingOriginal, setPlayingOriginal] = useState(false);
  const [playingEnhanced, setPlayingEnhanced] = useState(false);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const { toast } = useToast();

  const handleFile = useCallback((f: File) => {
    if (f.type.startsWith("audio/") || /\.(wav|mp3|flac|ogg|aac|m4a)$/i.test(f.name)) {
      setFile(f);
      setState("uploaded");
      setProgress(0);
      setEnhancedBlob(null);
      setErrorMsg("");
      if (enhancedUrl) URL.revokeObjectURL(enhancedUrl);
      setEnhancedUrl(null);
      if (originalUrl) URL.revokeObjectURL(originalUrl);
      setOriginalUrl(URL.createObjectURL(f));
    } else {
      toast({
        title: "Invalid file",
        description: "Please upload an audio file (WAV, MP3, FLAC, OGG, AAC).",
      });
    }
  }, [enhancedUrl, originalUrl, toast]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleEnhance = async () => {
    if (!file) return;
    setState("enhancing");
    setProgress(0);
    setErrorMsg("");

    // Simulate progress while waiting for API
    const progressInterval = setInterval(() => {
      setProgress((p) => {
        if (p >= 90) return 90;
        return p + Math.random() * 3 + 1;
      });
    }, 200);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/enhance`, {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Server error" }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      setProgress(95);

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      const pTime = response.headers.get("X-Processing-Time") || "";
      const rtfVal = response.headers.get("X-RTF") || "";

      setProgress(100);
      setEnhancedBlob(blob);
      if (enhancedUrl) URL.revokeObjectURL(enhancedUrl);
      setEnhancedUrl(url);
      setProcessingTime(pTime);
      setRtf(rtfVal);
      setState("done");

      toast({
        title: "Enhancement complete!",
        description: `Processed in ${pTime}s${rtfVal ? ` (RTF: ${rtfVal}x)` : ""}`,
      });
    } catch (err: unknown) {
      clearInterval(progressInterval);
      const message = err instanceof Error ? err.message : "Unknown error";
      setErrorMsg(message);
      setState("error");
      toast({
        title: "Enhancement failed",
        description: message,
      });
    }
  };

  const handleDownload = () => {
    if (!enhancedUrl || !file) return;
    const a = document.createElement("a");
    a.href = enhancedUrl;
    a.download = `enhanced_${file.name.replace(/\.[^.]+$/, "")}.wav`;
    a.click();
  };

  const toggleOriginalPlay = () => {
    const audio = originalAudioRef.current;
    if (!audio) return;
    if (playingOriginal) {
      audio.pause();
    } else {
      enhancedAudioRef.current?.pause();
      setPlayingEnhanced(false);
      audio.play();
    }
    setPlayingOriginal(!playingOriginal);
  };

  const toggleEnhancedPlay = () => {
    const audio = enhancedAudioRef.current;
    if (!audio) return;
    if (playingEnhanced) {
      audio.pause();
    } else {
      originalAudioRef.current?.pause();
      setPlayingOriginal(false);
      audio.play();
    }
    setPlayingEnhanced(!playingEnhanced);
  };

  const reset = () => {
    setFile(null);
    setState("idle");
    setProgress(0);
    setEnhancedBlob(null);
    setErrorMsg("");
    if (enhancedUrl) URL.revokeObjectURL(enhancedUrl);
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    setEnhancedUrl(null);
    setOriginalUrl(null);
    setPlayingOriginal(false);
    setPlayingEnhanced(false);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      {/* Hidden audio elements */}
      {originalUrl && (
        <audio
          ref={originalAudioRef}
          src={originalUrl}
          onEnded={() => setPlayingOriginal(false)}
          className="hidden"
        />
      )}
      {enhancedUrl && (
        <audio
          ref={enhancedAudioRef}
          src={enhancedUrl}
          onEnded={() => setPlayingEnhanced(false)}
          className="hidden"
        />
      )}

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4 overflow-hidden">
        <div className="absolute inset-0 gradient-radial pointer-events-none" />
        <div className="absolute inset-0 opacity-20 pointer-events-none">
          <img src={heroWave} alt="" className="w-full h-full object-cover" />
        </div>

        <div className="relative z-10 max-w-3xl mx-auto text-center space-y-6">
          <div className="inline-flex items-center gap-2 bg-primary/10 border border-primary/20 rounded-full px-4 py-1.5 text-sm text-primary font-medium">
            <Sparkles className="w-4 h-4" />
            Powered by Texutra World
          </div>

          <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-tight">
            <span className="text-gradient">Enhance</span>{" "}
            <span className="text-foreground">Your</span>
            <br />
            <span className="text-foreground">Audio Instantly</span>
          </h1>

          <p className="text-muted-foreground text-lg md:text-xl max-w-xl mx-auto leading-relaxed">
            Upload any audio file and let our AI transform it into studio-quality sound.
            Noise reduction, clarity boost, and professional mastering — all in seconds.
          </p>

          <a
            href="#enhance"
            className="inline-flex items-center gap-2 bg-primary text-primary-foreground px-8 py-3.5 rounded-xl font-semibold text-lg hover:bg-primary/90 transition-all glow-primary"
          >
            <Sparkles className="w-5 h-5" />
            Enhance Now — It's Free
          </a>
        </div>
      </section>

      {/* Stats */}
      <StatsSection />

      {/* Enhance Section */}
      <section id="enhance" className="py-20 px-4">
        <div className="max-w-lg mx-auto space-y-6 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground">
            <span className="text-gradient">Upload</span> & Enhance
          </h2>
          <p className="text-muted-foreground">Drop your audio file below to get started.</p>

          <div className="bg-card border border-border rounded-2xl p-8 space-y-6">
            {state === "idle" && (
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                className={cn(
                  "border-2 border-dashed rounded-xl p-10 cursor-pointer transition-all duration-300",
                  dragOver
                    ? "border-primary bg-primary/5 glow-primary"
                    : "border-border hover:border-primary/50 hover:bg-primary/5"
                )}
              >
                <Upload className="w-10 h-10 text-muted-foreground mx-auto mb-4" />
                <p className="text-foreground font-medium">Drop audio file here or click to browse</p>
                <p className="text-muted-foreground text-sm mt-1">MP3, WAV, FLAC, OGG, AAC supported</p>
                <input
                  ref={inputRef}
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFile(f);
                  }}
                />
              </div>
            )}

            {state !== "idle" && file && (
              <div className="space-y-6">
                {/* File info */}
                <div className="flex items-center gap-3 bg-secondary rounded-xl p-4">
                  <FileAudio className="w-8 h-8 text-primary shrink-0" />
                  <div className="text-left flex-1 min-w-0">
                    <p className="text-foreground font-medium truncate">{file.name}</p>
                    <p className="text-muted-foreground text-sm">{(file.size / (1024 * 1024)).toFixed(1)} MB</p>
                  </div>
                  {state === "uploaded" && (
                    <button onClick={reset} className="text-muted-foreground hover:text-foreground transition-colors">
                      <X className="w-5 h-5" />
                    </button>
                  )}
                </div>

                {/* Original audio playback */}
                {originalUrl && (state === "uploaded" || state === "done") && (
                  <div className="flex items-center gap-3 bg-secondary/50 rounded-xl p-3">
                    <button
                      onClick={toggleOriginalPlay}
                      className="w-8 h-8 rounded-full bg-muted flex items-center justify-center hover:bg-muted/80 transition-colors"
                    >
                      {playingOriginal ? (
                        <Pause className="w-4 h-4 text-foreground" />
                      ) : (
                        <Play className="w-4 h-4 text-foreground ml-0.5" />
                      )}
                    </button>
                    <span className="text-muted-foreground text-sm">Original Audio</span>
                  </div>
                )}

                <WaveformVisualizer isAnimating={state === "enhancing"} />

                {/* Enhancing progress */}
                {state === "enhancing" && (
                  <div className="space-y-2">
                    <div className="w-full h-1.5 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary rounded-full transition-all duration-200"
                        style={{ width: `${Math.min(progress, 100)}%` }}
                      />
                    </div>
                    <p className="text-muted-foreground text-sm">
                      {progress < 90 ? `Enhancing with AI... ${Math.round(progress)}%` : "Finalizing..."}
                    </p>
                  </div>
                )}

                {/* Upload state */}
                {state === "uploaded" && (
                  <Button
                    onClick={handleEnhance}
                    size="lg"
                    className="w-full bg-primary text-primary-foreground hover:bg-primary/90 glow-primary font-semibold text-base h-12"
                  >
                    <Sparkles className="w-5 h-5 mr-2" />
                    Enhance Audio
                  </Button>
                )}

                {/* Error state */}
                {state === "error" && (
                  <div className="space-y-3">
                    <p className="text-destructive font-medium text-sm">✕ Enhancement failed: {errorMsg}</p>
                    <Button
                      onClick={handleEnhance}
                      size="lg"
                      variant="outline"
                      className="w-full font-semibold text-base h-12"
                    >
                      <Sparkles className="w-5 h-5 mr-2" />
                      Try Again
                    </Button>
                    <button onClick={reset} className="text-muted-foreground hover:text-foreground text-sm transition-colors">
                      Upload a different file
                    </button>
                  </div>
                )}

                {/* Done state */}
                {state === "done" && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center">
                        <span className="text-primary text-sm">✓</span>
                      </div>
                      <p className="text-primary font-medium">Enhancement complete</p>
                    </div>

                    {processingTime && (
                      <p className="text-muted-foreground text-xs">
                        Processed in {processingTime}s
                        {rtf && parseFloat(rtf) < 1 && ` · ⚡ ${rtf}x faster than real-time`}
                      </p>
                    )}

                    {/* Enhanced audio playback */}
                    {enhancedUrl && (
                      <div className="flex items-center gap-3 bg-primary/10 rounded-xl p-3 border border-primary/20">
                        <button
                          onClick={toggleEnhancedPlay}
                          className="w-8 h-8 rounded-full bg-primary flex items-center justify-center hover:bg-primary/80 transition-colors"
                        >
                          {playingEnhanced ? (
                            <Pause className="w-4 h-4 text-primary-foreground" />
                          ) : (
                            <Play className="w-4 h-4 text-primary-foreground ml-0.5" />
                          )}
                        </button>
                        <span className="text-foreground text-sm font-medium">Enhanced Audio</span>
                      </div>
                    )}

                    <Button
                      onClick={handleDownload}
                      size="lg"
                      className="w-full bg-primary text-primary-foreground hover:bg-primary/90 glow-primary font-semibold text-base h-12"
                    >
                      <Download className="w-5 h-5 mr-2" />
                      Download Enhanced Audio
                    </Button>
                    <button onClick={reset} className="text-muted-foreground hover:text-foreground text-sm transition-colors">
                      Enhance another file
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Features */}
      <div id="features">
        <FeaturesSection />
      </div>

      {/* How It Works */}
      <div id="how-it-works">
        <HowItWorks />
      </div>

      {/* CTA */}
      <section className="py-20 px-4 text-center">
        <div className="max-w-2xl mx-auto space-y-6">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground">
            Ready to <span className="text-gradient">Transform</span> Your Audio?
          </h2>
          <p className="text-muted-foreground text-lg">
            Join thousands of creators who trust Texutra World for professional audio enhancement.
          </p>
          <a
            href="#enhance"
            className="inline-flex items-center gap-2 bg-primary text-primary-foreground px-8 py-3.5 rounded-xl font-semibold text-lg hover:bg-primary/90 transition-all glow-primary"
          >
            Get Started Free
          </a>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
