import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

export async function GET(
  request: NextRequest,
  { params }: { params: { filename: string } }
) {
  try {
    const filename = decodeURIComponent(params.filename);
    
    // Security: Only allow PDF files and sanitize filename
    if (!filename.endsWith('.pdf')) {
      return NextResponse.json({ error: 'Invalid file type' }, { status: 400 });
    }

    // Remove any path traversal attempts
    const sanitizedFilename = path.basename(filename);
    
    // Construct the path to the PDF file (assuming PDFs are in the pdfs directory at root)
    const filePath = path.join(process.cwd(), '..', 'pdfs', sanitizedFilename);
    
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: 'File not found' }, { status: 404 });
    }

    // Read the file
    const fileBuffer = fs.readFileSync(filePath);
    
    // Return the file with appropriate headers
    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `attachment; filename="${sanitizedFilename}"`,
        'Content-Length': fileBuffer.length.toString(),
      },
    });
  } catch (error) {
    console.error('Download error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}