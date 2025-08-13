import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    // Create response that redirects to home page
    const response = NextResponse.redirect(new URL("/", request.url));
    
    // Clear authentication cookies
    response.cookies.delete("auth_token");
    response.cookies.delete("user_email");
    response.cookies.delete("user_name");
    response.cookies.delete("is_admin");
    
    return response;
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to logout" },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  // Handle GET request for logout as well
  return POST(request);
}
