import { useState } from "react";
import { Link } from "react-router";
import { ChevronLeftIcon, EyeCloseIcon, EyeIcon } from "../../icons";
import Label from "../form/Label";
import Input from "../form/input/InputField";
import Checkbox from "../form/input/Checkbox";
import axios from "axios";

export default function SignUpForm() {
  const [showPassword, setShowPassword] = useState(false);
  const [isChecked, setIsChecked] = useState(false);
  const [formData, setFormData] = useState({
    username: "",
    fname: "",
    lname: "",
    email: "",
    password: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!isChecked) {
      setError("You must agree to Terms and Conditions");
      return;
    }

    setLoading(true);
    try {
      const payload = {
        username: formData.username.trim() || formData.email.split("@")[0],
        first_name: formData.fname.trim(),
        last_name: formData.lname.trim(),
        email: formData.email.trim(),
        password: formData.password,
      };

      const response = await axios.post(
        "http://127.0.0.1:8000/api/signup/",
        payload,
        { headers: { "Content-Type": "application/json" } }
      );

      console.log("User created:", response.data);
      setLoading(false);
      window.location.href = "/signin";
    } catch (err: any) {
      console.error(err);
      const respData = err.response?.data;
      let message = "Signup failed";
      if (respData) {
        if (respData.username) message = respData.username[0];
        else if (respData.email) message = respData.email[0];
        else if (respData.password) message = respData.password[0];
        else if (respData.detail) message = respData.detail;
      }
      setError(message);
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col flex-1 w-full overflow-y-auto lg:w-1/2 no-scrollbar">
      <div className="w-full max-w-md mx-auto mb-5 sm:pt-10">
        <Link
          to="/"
          className="inline-flex items-center text-sm text-gray-500 transition-colors hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
        >
          <ChevronLeftIcon className="size-5" />
          Back to dashboard
        </Link>
      </div>
      <div className="flex flex-col justify-center flex-1 w-full max-w-md mx-auto">
        <div>
          <div className="mb-5 sm:mb-8">
            <h1 className="mb-2 font-semibold text-gray-800 text-title-sm dark:text-white/90 sm:text-title-md">
              Sign Up
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Enter your details to create an account.
            </p>
          </div>
          <form onSubmit={handleSubmit}>
            {error && (
              <p className="mb-3 text-sm text-center text-red-500">{error}</p>
            )}
            <div className="space-y-5">
              <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
                <div>
                  <Label>First Name*</Label>
                  <Input
                    type="text"
                    name="fname"
                    placeholder="Enter first name"
                    value={formData.fname}
                    onChange={handleChange}
                    required
                  />
                </div>
                <div>
                  <Label>Last Name*</Label>
                  <Input
                    type="text"
                    name="lname"
                    placeholder="Enter last name"
                    value={formData.lname}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
              <div>
                <Label>Username (optional)</Label>
                <Input
                  type="text"
                  name="username"
                  placeholder="Enter username"
                  value={formData.username}
                  onChange={handleChange}
                />
              </div>
              <div>
                <Label>Email*</Label>
                <Input
                  type="email"
                  name="email"
                  placeholder="Enter email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                />
              </div>
              <div>
                <Label>Password*</Label>
                <div className="relative">
                  <Input
                    placeholder="Enter password"
                    type={showPassword ? "text" : "password"}
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    required
                  />
                  <span
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute z-30 -translate-y-1/2 cursor-pointer right-4 top-1/2"
                  >
                    {showPassword ? (
                      <EyeIcon className="size-5 text-gray-500" />
                    ) : (
                      <EyeCloseIcon className="size-5 text-gray-500" />
                    )}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Checkbox
                  className="w-5 h-5"
                  checked={isChecked}
                  onChange={setIsChecked}
                />
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  By creating an account you agree to{" "}
                  <span className="text-gray-800 dark:text-white/90">
                    Terms
                  </span>{" "}
                  and{" "}
                  <span className="text-gray-800 dark:text-white/90">
                    Privacy Policy
                  </span>
                </p>
              </div>
              <div>
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full px-4 py-3 text-sm font-medium text-white bg-brand-500 rounded-lg hover:bg-brand-600 disabled:opacity-60"
                >
                  {loading ? "Signing Up..." : "Sign Up"}
                </button>
              </div>
            </div>
          </form>
          <div className="mt-5 text-center text-sm text-gray-700 dark:text-gray-400">
            Already have an account?{" "}
            <Link
              to="/signin"
              className="text-brand-500 hover:text-brand-600 dark:text-brand-400"
            >
              Sign In
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
