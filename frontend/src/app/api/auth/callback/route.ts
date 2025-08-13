import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const ticket = searchParams.get("ticket");

  console.log("CAS callback received, ticket:", ticket ? `${ticket.substring(0, 10)}...` : "none");

  if (!ticket) {
    console.error("No ticket received in callback");
    return NextResponse.redirect(new URL("/admin?error=no_ticket", request.url));
  }

  try {
    // Validate the ticket with the backend
    const serviceUrl = `${request.nextUrl.origin}/api/auth/callback`;
    console.log("Validating ticket with service URL:", serviceUrl);
    
    const response = await fetch(`${BACKEND_URL}/auth/validate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ticket,
        service: serviceUrl,
      }),
    });

    console.log("Validation response status:", response.status);
    
    if (response.ok) {
      const userData = await response.json();
      console.log("Validation successful, user data:", {
        email: userData.user?.email,
        is_admin: userData.user?.is_admin
      });
      
      // Create a response that redirects to admin page
      const redirectResponse = NextResponse.redirect(new URL("/admin", request.url));
      
      // Set authentication data in cookies
      if (userData.user) {
        redirectResponse.cookies.set("user_email", userData.user.email, {
          httpOnly: true,
          secure: process.env.NODE_ENV === "production",
          maxAge: 86400, // 24 hours
          path: "/",
        });
        
        redirectResponse.cookies.set("user_name", userData.user.full_name || userData.user.username, {
          httpOnly: true,
          secure: process.env.NODE_ENV === "production",
          maxAge: 86400, // 24 hours
          path: "/",
        });
        
        redirectResponse.cookies.set("is_admin", userData.user.is_admin.toString(), {
          httpOnly: true,
          secure: process.env.NODE_ENV === "production",
          maxAge: 86400, // 24 hours
          path: "/",
        });
        
        redirectResponse.cookies.set("auth_token", "authenticated", {
          httpOnly: true,
          secure: process.env.NODE_ENV === "production",
          maxAge: 86400, // 24 hours
          path: "/",
        });
        
        console.log("Cookies set, redirecting to /admin");
      }

      return redirectResponse;
    } else {
      const errorData = await response.text();
      console.error("Validation failed:", response.status, errorData);
      return NextResponse.redirect(new URL("/admin?error=auth_failed", request.url));
    }
  } catch (error) {
    console.error("CAS validation error:", error);
    return NextResponse.redirect(new URL("/admin?error=validation_error", request.url));
  }
}
