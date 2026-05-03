import { Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Sparkles, Hourglass, XCircle } from "lucide-react";
import type { Application } from "@/lib/api";

type Props = {
  application: Application;
};

export function WaitingScreen({ application }: Props) {
  const { stage } = application;

  let Icon = Hourglass;
  let heading = "Application not available yet";
  let body =
    "We couldn't open the interview for this application. Head back to the dashboard to see what's next.";

  if (stage === "applied") {
    Icon = Hourglass;
    heading = "Your application is under review";
    body =
      "We've sent your application to the team. You'll get a notification as soon as a recruiter approves it — then this page will unlock automatically.";
  } else if (stage === "rejected") {
    Icon = XCircle;
    heading = "This application is closed";
    body =
      "Thanks for your interest. The team isn't moving forward with this application. You can browse other open roles whenever you're ready.";
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-ink text-white">
      <div className="absolute inset-0 bg-violet-grad opacity-20" />
      <header className="relative flex items-center justify-between border-b border-white/10 px-6 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-lg bg-white/10">
            <Sparkles className="h-3.5 w-3.5" />
          </div>
          <div>
            <div className="text-xs text-white/60">AI Interview Room</div>
            <div className="text-sm font-semibold">Awaiting recruiter</div>
          </div>
        </div>
        <Link to="/c" className="text-xs text-white/60 hover:text-white">
          Exit
        </Link>
      </header>
      <div className="relative flex flex-1 items-center justify-center px-6">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md rounded-2xl border border-white/10 bg-white/5 p-8 text-center backdrop-blur"
        >
          <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-white/10">
            <Icon className="h-6 w-6" />
          </div>
          <h2 className="mt-4 font-display text-2xl">{heading}</h2>
          <p className="mt-2 text-sm text-white/70">{body}</p>
          <Link
            to="/c/applications"
            className="mt-6 inline-flex items-center justify-center rounded-full bg-white px-4 py-2 text-sm font-semibold text-ink"
          >
            Track my application
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
