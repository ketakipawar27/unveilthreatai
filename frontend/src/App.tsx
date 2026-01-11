import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import SignIn from "./pages/AuthPages/SignIn";
import SignUp from "./pages/AuthPages/SignUp";
import NotFound from "./pages/OtherPage/NotFound";
import UserProfiles from "./pages/UserProfiles";
import Videos from "./pages/UiElements/Videos";
import Images from "./pages/UiElements/Images";
import Alerts from "./pages/UiElements/Alerts";
import Badges from "./pages/UiElements/Badges";
import Avatars from "./pages/UiElements/Avatars";
import Buttons from "./pages/UiElements/Buttons";
import FormElements from "./pages/Forms/FormElements";
import Blank from "./pages/Blank";
import AppLayout from "./layout/AppLayout";
import { ScrollToTop } from "./components/common/ScrollToTop";
import Home from "./pages/Dashboard/Home";
import LandingPage from "./pages/landing/LandingPage";
import AllSimulations from "./pages/AllSimulations";  // ✅ import
import AnalyzePage from "./pages/AnalyzePage";

export default function App() {
  return (
    <Router>
      <ScrollToTop />
      <Routes>
        {/* Landing Page as default first page */}
        <Route path="/" element={<LandingPage />} />

        {/* Dashboard Layout */}
        <Route element={<AppLayout />}>
          <Route path="/home" element={<Home />} />

          {/* Other Pages */}
          <Route path="/profile" element={<UserProfiles />} />
          <Route path="/blank" element={<Blank />} />

          {/* Forms */}
          <Route path="/form-elements" element={<FormElements />} />

          {/* UI Elements */}
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/avatars" element={<Avatars />} />
          <Route path="/badge" element={<Badges />} />
          <Route path="/buttons" element={<Buttons />} />
          <Route path="/images" element={<Images />} />
          <Route path="/videos" element={<Videos />} />

          {/* ✅ All Simulations Page */}
          <Route path="/simulations/all" element={<AllSimulations />} />
        </Route>

        {/* Auth Pages */}
        <Route path="/signin" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />

        {/* Fallback Route */}
        <Route path="*" element={<NotFound />} />


         <Route path="/analyze" element={<AnalyzePage />} />
      </Routes>
    </Router>
  );
}
