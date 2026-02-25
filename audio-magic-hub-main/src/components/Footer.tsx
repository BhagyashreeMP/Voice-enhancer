const Footer = () => {
  return (
    <footer className="py-10 px-4 border-t border-border">
      <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
            <span className="text-primary font-bold text-sm">T</span>
          </div>
          <span className="text-foreground font-semibold">Texutra World</span>
        </div>
        <p className="text-muted-foreground text-sm">
          © {new Date().getFullYear()} Texutra World. All rights reserved.
        </p>
        <div className="flex gap-6 text-muted-foreground text-sm">
          <a href="#" className="hover:text-foreground transition-colors">Privacy</a>
          <a href="#" className="hover:text-foreground transition-colors">Terms</a>
          <a href="#" className="hover:text-foreground transition-colors">Contact</a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
