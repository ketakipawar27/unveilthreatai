import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import PageMeta from "../../components/common/PageMeta";

// Variants for staggered container animations
const container = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { staggerChildren: 0.2 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

export default function LandingPage() {
  return (
    <>
      <PageMeta
        title="Security Dashboard"
        description="Landing page for Security Dashboard project"
      />

      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col">

        {/* Hero Section */}
        <header className="bg-white dark:bg-gray-800 shadow-md">
          <motion.div
            className="max-w-7xl mx-auto px-6 py-20 text-center"
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Logo */}
            <Link to="/" className="mb-4 flex justify-center">
              <motion.img
                className="dark:hidden h-12"
                src="./images/logo/logo.svg"
                alt="Logo"
                whileHover={{ scale: 1.1 }}
                transition={{ type: "spring", stiffness: 300 }}
              />
              <motion.img
                className="hidden dark:block h-12"
                src="./images/logo/logo-dark.svg"
                alt="Logo"
                whileHover={{ scale: 1.1 }}
                transition={{ type: "spring", stiffness: 300 }}
              />
            </Link>

            <motion.h1
              className="text-5xl font-bold text-gray-900 dark:text-white mb-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.8 }}
            >
              Security Dashboard
            </motion.h1>

            <motion.p
              className="text-gray-600 dark:text-gray-300 text-lg mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.8 }}
            >
              Analyze and visualize your file security with ease. Track metadata exposure, file risks, and generate comprehensive reports.
            </motion.p>

            <motion.a
              href="/signin"
              className="inline-block bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-6 py-3 rounded-xl transition"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Get Started
            </motion.a>
          </motion.div>
        </header>

        {/* Features Section */}
        <motion.section
          className="max-w-7xl mx-auto px-6 py-16 grid grid-cols-1 md:grid-cols-3 gap-6"
          variants={container}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          {[
            {
              title: "File Analysis",
              description:
                "Upload files and instantly get insights on metadata and sensitive information.",
            },
            {
              title: "Risk Meter",
              description:
                "Quickly assess how much sensitive data is exposed and track file security risks.",
            },
            {
              title: "Reports",
              description:
                "Generate detailed reports for each file and monitor security over time.",
            },
          ].map((feature) => (
            <motion.div
              key={feature.title}
              className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-white/[0.03]"
              variants={item}
              whileHover={{ scale: 1.03 }}
            >
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90 mb-2">
                {feature.title}
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.section>

        {/* How It Works Section */}
        <motion.section
          className="bg-gray-100 dark:bg-gray-800 py-16"
          variants={container}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <div className="max-w-7xl mx-auto px-6 text-center">
            <motion.h2
              className="text-3xl font-bold text-gray-900 dark:text-white mb-8"
              variants={item}
            >
              How It Works
            </motion.h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                {
                  title: "Upload File",
                  description:
                    "Upload any file and start analyzing metadata and sensitive information instantly.",
                },
                {
                  title: "View Risk Metrics",
                  description:
                    "See risk levels, exposure percentages, and detailed analysis at a glance.",
                },
                {
                  title: "Generate Reports",
                  description:
                    "Export detailed reports or view full dashboards for complete insights.",
                },
              ].map((step) => (
                <motion.div
                  key={step.title}
                  className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-white/[0.03]"
                  variants={item}
                  whileHover={{ scale: 1.03 }}
                >
                  <h4 className="text-lg font-semibold text-gray-800 dark:text-white/90 mb-2">
                    {step.title}
                  </h4>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {step.description}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Dashboard Preview Section */}
        <motion.section
          className="max-w-7xl mx-auto px-6 py-16"
          variants={container}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <motion.h2
            className="text-3xl font-bold text-gray-900 dark:text-white text-center mb-10"
            variants={item}
          >
            Dashboard Preview
          </motion.h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {["/images/dashboard-1.png", "/images/dashboard-2.png"].map((img, idx) => (
              <motion.div
                key={img}
                className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-white/[0.03]"
                variants={item}
                whileHover={{ scale: 1.02 }}
              >
                <img src={img} alt={`Dashboard ${idx + 1}`} className="rounded-xl" />
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Testimonials Section */}
        <motion.section
          className="bg-gray-100 dark:bg-gray-800 py-16"
          variants={container}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <div className="max-w-7xl mx-auto px-6 text-center">
            <motion.h2
              className="text-3xl font-bold text-gray-900 dark:text-white mb-8"
              variants={item}
            >
              What Users Say
            </motion.h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                {
                  text: `"This dashboard helped us identify file risks in minutes. The UI is clean and intuitive."`,
                  author: "Alex Johnson",
                },
                {
                  text: `"The metadata insights are amazing! I can now track sensitive information easily."`,
                  author: "Maria Garcia",
                },
                {
                  text: `"Reports are detailed and the risk meter gives me a quick overview of exposure levels."`,
                  author: "James Lee",
                },
              ].map((testimonial) => (
                <motion.div
                  key={testimonial.author}
                  className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-white/[0.03]"
                  variants={item}
                  whileHover={{ scale: 1.02 }}
                >
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">{testimonial.text}</p>
                  <span className="font-medium text-gray-800 dark:text-white/90">{testimonial.author}</span>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Call to Action Section */}
        <motion.section
          className="py-16 text-center"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <motion.h2
            className="text-3xl font-bold text-gray-900 dark:text-white mb-4"
          >
            Ready to Start?
          </motion.h2>
          <motion.p className="text-gray-600 dark:text-gray-300 mb-6">
            Upload your files and start analyzing them securely with our dashboard.
          </motion.p>
          <motion.a
            href="/signin"
            className="inline-block bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-6 py-3 rounded-xl transition"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Get Started
          </motion.a>
        </motion.section>

        {/* Footer */}
        <footer className="bg-white dark:bg-gray-800 shadow-inner mt-auto">
          <div className="max-w-7xl mx-auto px-6 py-6 text-center text-gray-600 dark:text-gray-400">
            &copy; {new Date().getFullYear()} Security Dashboard. All rights reserved.
          </div>
        </footer>
      </div>
    </>
  );
}
